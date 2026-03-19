package ocr

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"os"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

/*
*
ppocr 配置
*/
type PaddleOCRConfig struct {
	// 必填参数
	OnnxRuntimeLibPath string // onnxruntime.dll (或 .so, .dylib) 的路径
	DetModelPath       string // det.onnx (检测模型) 的路径
	RecModelPath       string // rec.onnx (识别模型) 的路径
	ClsModelPath       string // cls.onnx (方向模型) 的路径
	DictPath           string // dict.txt (字典) 的路径

	// 可选参数
	UseCuda            bool   // (可选) 是否启用 CUDA
	NumThreads         int    // (可选) ONNX 线程数, 默认由CPU核心数决定
	RecModelNumClasses int    // (可选) 识别模型类别数, 默认 18385
	RecBatchSize       int    // (可选) 识别模型批数, 默认 6个
	RecImageShape      [4]int //(可选) 识别模型的缩放尺寸默认 {3（通道）, 48（高度）, 320（最小宽度）,3200（最大宽度）}

	ClsCropSize [2]int //  (可选) 识方向模型裁剪大小 默认{224, 224}
	ClsMap      [4]int // (可选)方向模型的旋转角度  默认（0,90,180,270）

	HeatmapThreshold float32 // 像素点的阈值, 默认 0.3
	BoxThresh        float32 // 文本检测框阈值, 默认 0.6
	MinSize          float32 // 距离图片最小范围
	UnclipRatio      float32 // 1.5
	// 三个检测模型归一化 参数
	Scale        float32
	Mean         []float32
	Std          []float32
	MaxSideLimit int // 最大缩放比例  默认4000
	LimitSideLen int // 限制缩放比例 默认64

	UseLog bool // 是否开启日志
}
type OcrResult struct {
	RecResult []*RecResult
	DetResult []*DetResult
}

/*
*
默认配置
*/
func NewDefaultPaddleOCRConfig(
	onnxRuntimeLibPath string,
	detModelPath string,
	recModelPath string,
	clsModelPath string,
	dictPath string,
	useCuda bool,
	useLog bool,

) *PaddleOCRConfig {
	config := &PaddleOCRConfig{
		OnnxRuntimeLibPath: onnxRuntimeLibPath,
		DetModelPath:       detModelPath,
		RecModelPath:       recModelPath,
		ClsModelPath:       clsModelPath,
		DictPath:           dictPath,
		MaxSideLimit:       960,
		LimitSideLen:       64,
		HeatmapThreshold:   0.3,
		BoxThresh:          0.6,
		MinSize:            3,
		RecBatchSize:       6,
		RecModelNumClasses: 18385,
		Scale:              float32(1.0 / 255.0),
		Mean:               []float32{0.485, 0.456, 0.406},
		Std:                []float32{0.229, 0.224, 0.225},
		RecImageShape:      [4]int{3, 48, 320, 3200},
		ClsCropSize:        [2]int{224, 224},
		ClsMap:             [4]int{0, 90, 180, 270},
		UseCuda:            useCuda,
		NumThreads:         10,
		UnclipRatio:        1.5,
		UseLog:             useLog,
	}

	return config
}

// pp ocr的回话
type PaddleOCRSession struct {
	clsSession *ClsOnnxSession  // 方向
	detSession *DetOnnxSession  // 检测
	recSession *RecOnnxSession  // 识别
	config     *PaddleOCRConfig // 方向

}

func (session *PaddleOCRSession) rotateImage(img *gocv.Mat, clsResult *ClsResult) (*gocv.Mat, error) {

	if clsResult.Label < 0 || clsResult.Label >= 360 {
		return nil, fmt.Errorf("angle should be in range [0, 360)")
	}
	h := img.Rows()
	w := img.Cols()
	result := clsResult

	angle := result.Label

	centerW := (w / 2)
	centerH := (h / 2)
	scale := 1.0

	mat := gocv.GetRotationMatrix2D(image.Pt(centerW, centerH), float64(angle), scale)

	cos := math.Abs(mat.GetDoubleAt(0, 0))
	sin := math.Abs(mat.GetDoubleAt(0, 1))
	// 1707
	newW := int((float64(h) * sin) + (float64(w) * cos))
	newH := int((float64(h) * cos) + (float64(w) * sin)) //1280

	mat.SetDoubleAt(0, 2, mat.GetDoubleAt(0, 2)+(float64(newW-w)/2))
	mat.SetDoubleAt(1, 2, mat.GetDoubleAt(1, 2)+(float64(newH-h)/2))

	rotated := gocv.NewMat()

	err := gocv.WarpAffineWithParams(*img,
		&rotated, mat,
		image.Pt(newW, newH), // 输出尺寸
		gocv.InterpolationCubic,
		gocv.BorderConstant,
		color.RGBA{},
	)

	if err != nil {
		return nil, err
	}

	//w1 := gocv.NewWindow("image")
	//w1.ResizeWindow(newW, newH)
	//w1.IMShow(rotated)
	//w1.WaitKey(0)

	return &rotated, nil
}

// RunOCR 对图像执行检测和识别
func (session *PaddleOCRSession) RunOCR(imagePath string) (*OcrResult, error) {

	image := gocv.IMRead(imagePath, gocv.IMReadColor)

	if image.Empty() {
		return nil, fmt.Errorf("image is empty")
	}
	img := &image
	defer img.Close()
	fileName := filepath.Base(imagePath)
	// 1.方向
	clsResult, err := session.clsSession.Run(img)
	if err != nil {
		return nil, err
	}

	rotateImage := img
	// 不等于0 证明需要旋转图片
	if clsResult.Index != 0 {
		// 2 旋转图片
		rotateImage, err = session.rotateImage(img, clsResult)
		if err != nil {
			return nil, err
		}
		defer rotateImage.Close()
	}

	// 3.检测
	detResult, err := session.detSession.Run(rotateImage)
	if err != nil {
		return nil, err
	}
	if len(detResult) == 0 {
		return &OcrResult{}, nil
	}
	// 4. 图片裁剪
	cropImages := getCropImages(rotateImage, detResult)

	if session.config.UseLog {
		pts := gocv.NewPointsVector()
		for i := 0; i < len(detResult); i++ {
			det := detResult[i]
			pt := gocv.NewPointVectorFromPoints(det.Points) // []image.Point
			pts.Append(pt)
			pt.Close()

		}
		gocv.Polylines(rotateImage, pts, true, color.RGBA{255, 0, 0, 0}, 2)
		gocv.IMWrite("det_result_"+fileName, *rotateImage)
		log.Printf("det_result_%s", fileName)
	}

	// 5.识别
	recResult := session.recSession.Run(cropImages)

	return &OcrResult{recResult, detResult}, nil
}

func getCropImages(img *gocv.Mat, scores []*DetResult) []*gocv.Mat {
	scoresSize := len(scores)
	cropMats := make([]*gocv.Mat, scoresSize)
	for i := 0; i < scoresSize; i++ {
		cropMats[i] = getMinAreaRectCrop(img, scores[i])
	}
	return cropMats
}

// Destroy 释放所有引擎相关的资源
func (session *PaddleOCRSession) Destroy() {
	if session.clsSession != nil {
		session.clsSession.OnnxSession.Destroy()
	}

	if session.detSession != nil {
		session.detSession.OnnxSession.Destroy()
	}
	if session.recSession != nil {
		session.recSession.OnnxSession.Destroy()
	}

}

func NewPaddleOCRSession(config *PaddleOCRConfig) (*PaddleOCRSession, error) {
	// 初始化 ONNX Runtime
	if config.OnnxRuntimeLibPath == "" {
		return nil, fmt.Errorf("OnnxRuntimeLibPath is required")
	}
	ort.SetSharedLibraryPath(config.OnnxRuntimeLibPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("failed to initialize ONNX Runtime environment: %w", err)
	}

	// 检查和设置默认值
	if config.DetModelPath == "" || config.RecModelPath == "" || config.DictPath == "" {
		return nil, fmt.Errorf("模型路径和字典路径不能为空")
	}

	// 创建会话选项 (设置线程)
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	if config.NumThreads > 0 {
		// 设置线程
		if err := options.SetIntraOpNumThreads(config.NumThreads); err != nil {
			return nil, err
		}
	}

	// 启用CUDA
	if config.UseCuda {
		cudaOptions, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("failed to create CUDA provider options: %w", err)
		}
		defer cudaOptions.Destroy()
		if err := options.AppendExecutionProviderCUDA(cudaOptions); err != nil {
			return nil, fmt.Errorf("failed to append CUDA execution provider: %w", err)
		}
	}

	// 创建方向模型
	clsSessionInternal, err := ort.NewDynamicAdvancedSession(
		config.ClsModelPath,
		[]string{"x"},            // 输入节点名
		[]string{"fetch_name_0"}, // 输出节点名
		options,
	)

	if err != nil {
		clsSessionInternal.Destroy()
		return nil, fmt.Errorf("failed to create classification session: %w", err)
	}

	// 创建检测模型
	detSessionInternal, err := ort.NewDynamicAdvancedSession(
		config.DetModelPath,
		[]string{"x"},            // 输入节点名
		[]string{"fetch_name_0"}, // 输出节点名
		options,
	)
	if err != nil {
		detSessionInternal.Destroy()
		return nil, fmt.Errorf("failed to create detection session: %w", err)
	}

	// 创建识别会话
	recSessionInternal, err := ort.NewDynamicAdvancedSession(
		config.RecModelPath,
		[]string{"x"},            // 输入节点名
		[]string{"fetch_name_0"}, // 输出节点名
		options,
	)
	if err != nil {
		recSessionInternal.Destroy() // 清理已创建的 det session
		return nil, fmt.Errorf("failed to create recognition session: %w", err)
	}

	// 加载字典
	dict, err := loadDictFile(config.DictPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load dictionary file: %w", err)
	}
	detSession := &DetOnnxSession{detSessionInternal, config}
	recSession := &RecOnnxSession{recSessionInternal, config, dict}
	clsSession := &ClsOnnxSession{
		OnnxSession: clsSessionInternal,
		config:      config,
	}

	session := &PaddleOCRSession{
		clsSession,
		detSession,
		recSession,
		config}
	return session, nil
}

func loadDictFile(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open dictionary file (%s): %w", path, err)
	}
	defer file.Close()
	var lines []string
	lines = append(lines, "blank")
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("scan dictionary file (%s): %w", path, err)
	}
	lines = append(lines, " ")
	return lines, nil
}

func HWCToCHW(resizedImage *gocv.Mat) []float32 {
	// 转换chw
	h := resizedImage.Rows()
	w := resizedImage.Cols()
	c := resizedImage.Channels()
	chw := make([]float32, c*h*w)
	data, err := resizedImage.DataPtrFloat32()
	if err != nil {
		panic(err)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for ch := 0; ch < c; ch++ {
				hwcIndex := (y*w+x)*c + ch
				chwIndex := ch*h*w + y*w + x
				chw[chwIndex] = data[hwcIndex]
			}
		}
	}
	return chw
}
