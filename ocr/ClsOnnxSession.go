package ocr

import (
	"fmt"
	"image"
	"math"

	"github.com/weihuanwan/paddleocr-go/common"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

type ClsOnnxSession struct {
	OnnxSession *ort.DynamicAdvancedSession

	Labels   [4]int // 标签字典
	CropSize [2]int // 截图大小（默认，224，224）
	Alpha    [3]float32
	Beta     [3]float32
}

func NewDefaultTableClsOnnxSession(onnxSession *ort.DynamicAdvancedSession) *ClsOnnxSession {
	alpha, beta := common.GetNormalizeAlphaBeta()
	return &ClsOnnxSession{
		OnnxSession: onnxSession,
		Labels:      [4]int{0, 90, 180, 270},
		CropSize:    [2]int{224, 224},
		Alpha:       alpha,
		Beta:        beta,
	}
}

func NewTableClsOnnxSession(onnxSession *ort.DynamicAdvancedSession, alpha [3]float32, beta [3]float32) *ClsOnnxSession {
	return &ClsOnnxSession{
		OnnxSession: onnxSession,
		Labels:      [4]int{0, 90, 180, 270},
		CropSize:    [2]int{224, 224},
		Alpha:       alpha,
		Beta:        beta,
	}
}

type ClsResult struct {
	Label int
	Index int
	Score float32
}

func (cls *ClsOnnxSession) Run(image *gocv.Mat) (*ClsResult, error) {

	resizeImage, err := cls.resize(image)

	if err != nil {
		return nil, err
	}

	defer resizeImage.Close()

	cropImage := cls.crop(resizeImage)
	defer cropImage.Close()

	normalizeImage, err := common.Normalize(cropImage, cls.Alpha, cls.Beta)
	if err != nil {
		return nil, err
	}
	defer normalizeImage.Close()

	imageCHW := common.HWCToCHW(normalizeImage)

	// 描述
	shape := ort.NewShape(1, 3, int64(normalizeImage.Rows()), int64(normalizeImage.Cols()))
	// 创建张量
	clsInputTensor, err := ort.NewTensor(shape, imageCHW)
	if err != nil {

		return nil, fmt.Errorf("ClsOnnxSession create  input tensor error", err.Error())
	}
	defer clsInputTensor.Destroy()

	clsOutputShape := ort.NewShape(1, 4)
	clsOutputTensor, err := ort.NewEmptyTensor[float32](clsOutputShape)

	if err != nil {
		return nil, fmt.Errorf("ClsOnnxSession create  output tensor error", err.Error())
	}
	defer clsOutputTensor.Destroy()
	// 检测（核心）
	err = cls.OnnxSession.Run([]ort.Value{clsInputTensor}, []ort.Value{clsOutputTensor})

	if err != nil {
		return nil, fmt.Errorf("ClsOnnxSession run  error ", err)
	}

	// 解析输出
	output := clsOutputTensor.GetData()

	maxIdx := 0
	maxScore := float32(-1)

	for i, v := range output {
		if v > maxScore {
			maxScore = v
			maxIdx = i
		}
	}
	label := cls.Labels[maxIdx]

	return &ClsResult{
		Label: label,
		Score: maxScore,
		Index: maxIdx,
	}, nil
}

func (cls *ClsOnnxSession) resize(originImage *gocv.Mat) (*gocv.Mat, error) {

	rgbOriginImage := gocv.NewMat()
	// bgr 转换 rgb
	err := gocv.CvtColor(*originImage, &rgbOriginImage, gocv.ColorBGRToRGB)

	if err != nil {
		return nil, fmt.Errorf("ClsOnnxSession Color BGR To RGB Error %w", err)
	}

	h := originImage.Rows()
	w := originImage.Cols()

	scale := float64(256) / float64(min(h, w))

	resizeH := int(math.Round(float64(h) * scale))
	resizeW := int(math.Round(float64(w) * scale))

	resizeImage := gocv.NewMat()

	err = gocv.Resize(rgbOriginImage,
		&resizeImage, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)

	if err != nil {
		return nil, fmt.Errorf("cls resize error %w", err)
	}

	return &resizeImage, nil
}
func (cls *ClsOnnxSession) crop(resizeImage *gocv.Mat) *gocv.Mat {

	h := resizeImage.Rows()
	w := resizeImage.Cols()

	cw, ch := cls.CropSize[0], cls.CropSize[1]

	x1 := max(0, (w-cw)/2)
	y1 := max(0, (h-ch)/2)

	x2 := min(w, x1+cw)
	y2 := min(h, y1+ch)

	croppedRegion := resizeImage.Region(image.Rect(x1, y1, x2, y2))
	return &croppedRegion
}
