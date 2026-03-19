package ocr

import (
	"fmt"
	"image"
	"log"
	"math"

	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

type ClsOnnxSession struct {
	OnnxSession *ort.DynamicAdvancedSession
	config      *PaddleOCRConfig
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

	normalizeImage, err := cls.normalize(cropImage)
	if err != nil {
		return nil, err
	}
	defer normalizeImage.Close()

	imageCHW := HWCToCHW(normalizeImage)

	// 描述
	shape := ort.NewShape(1, 3, int64(normalizeImage.Rows()), int64(normalizeImage.Cols()))
	// 创建张量
	clsInputTensor, err := ort.NewTensor(shape, imageCHW)
	if err != nil {

		return nil, fmt.Errorf("create rec input tensor error", err.Error())
	}
	defer clsInputTensor.Destroy()

	clsOutputShape := ort.NewShape(1, 4)
	clsOutputTensor, err := ort.NewEmptyTensor[float32](clsOutputShape)

	if err != nil {
		return nil, fmt.Errorf("create det output tensor error", err.Error())
	}
	defer clsOutputTensor.Destroy()
	// 检测（核心）
	err = cls.OnnxSession.Run([]ort.Value{clsInputTensor}, []ort.Value{clsOutputTensor})

	if err != nil {
		return nil, fmt.Errorf("run cls OnnxSession error ", err)
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
	label := cls.config.ClsMap[maxIdx]
	if cls.config.UseLog {
		log.Printf("cls label %d score %f", label, maxScore)
	}

	return &ClsResult{
		Label: label,
		Score: maxScore,
		Index: maxIdx,
	}, nil
}

func (cls *ClsOnnxSession) resize(originImage *gocv.Mat) (*gocv.Mat, error) {

	rgbOriginImage := gocv.NewMat()
	// bgr 转换 rgb
	gocv.CvtColor(*originImage, &rgbOriginImage, gocv.ColorBGRToRGB)

	h := originImage.Rows()
	w := originImage.Cols()

	scale := float64(256) / float64(min(h, w))

	resizeH := int(math.Round(float64(h) * scale))
	resizeW := int(math.Round(float64(w) * scale))

	resizeImage := gocv.NewMat()

	err := gocv.Resize(rgbOriginImage,
		&resizeImage, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)

	if err != nil {
		return nil, fmt.Errorf("cls resize error")
	}

	return &resizeImage, nil
}
func (cls *ClsOnnxSession) crop(resizeImage *gocv.Mat) *gocv.Mat {

	h := resizeImage.Rows()
	w := resizeImage.Cols()

	cw, ch := cls.config.ClsCropSize[0], cls.config.ClsCropSize[1]

	x1 := max(0, (w-cw)/2)
	y1 := max(0, (h-ch)/2)

	x2 := min(w, x1+cw)
	y2 := min(h, y1+ch)

	// 使用Region方法裁剪（推荐）
	croppedRegion := resizeImage.Region(image.Rect(x1, y1, x2, y2))
	return &croppedRegion
}

// 归一化处理
func (cls *ClsOnnxSession) normalize(resizedImage *gocv.Mat) (*gocv.Mat, error) {
	c := resizedImage.Channels()
	// 获取rgb
	gbrSplit := gocv.Split(*resizedImage)

	scale := cls.config.Scale
	mean := cls.config.Mean
	std := cls.config.Std
	var alpha [3]float32
	var beta [3]float32

	for i := 0; i < c; i++ {
		alpha[i] = scale / std[i]
		beta[i] = -mean[i] / std[i]
	}
	for i := range c {
		cpMat := gocv.NewMat()
		old := gbrSplit[i]
		//转换 32 位
		err := old.ConvertTo(&cpMat, gocv.MatTypeCV32F)
		if err != nil {
			return nil, fmt.Errorf("cls normalize convert to 32f error")
		}
		gbrSplit[i] = cpMat
		// 该通道乘以
		gbrSplit[i].MultiplyFloat(alpha[i])
		// 在加上
		gbrSplit[i].AddFloat(beta[i])

	}
	result := gocv.NewMat()

	err := gocv.Merge(gbrSplit, &result)
	if err != nil {
		return nil, fmt.Errorf("cls normalize merge error")
	}
	return &result, nil
}
