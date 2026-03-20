package ocr

import (
	"fmt"
	"image"

	"github.com/weihuanwan/paddleocr-go/common"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

/*
*版面区域检测模块
 */
type DocLayoutPlusLSession struct {
	OnnxSession *ort.DynamicAdvancedSession
	Alpha       [3]float32
	Beta        [3]float32
}

func NewDocLayoutPlusLSession(onnxSession *ort.DynamicAdvancedSession) *DocLayoutPlusLSession {
	scale := float32(1.0 / 255.0)
	mean := []float32{0, 0, 0}
	std := []float32{1, 1, 1}
	var alpha [3]float32
	var beta [3]float32

	for i := 0; i < 3; i++ {
		alpha[i] = scale / std[i]
		beta[i] = -mean[i] / std[i]
	}

	return &DocLayoutPlusLSession{
		onnxSession,
		alpha,
		beta,
	}
}

func (docLayout *DocLayoutPlusLSession) Run(originImage *gocv.Mat) ([]*DetResult, error) {
	// 缩放
	resizedImage, scale, err := docLayout.resize(originImage)
	if err != nil {
		return nil, err
	}

	defer resizedImage.Close()
	// 归一化 # [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
	//# [-0.0, -0.0, -0.0]
	imageNormalize, err := common.Normalize(resizedImage, docLayout.Alpha, docLayout.Beta)

	if err != nil {
		return nil, err
	}

	defer imageNormalize.Close()

	// 转换 chw
	imageCHW := common.HWCToCHW(imageNormalize)

	// 1.
	imageTensor, err := ort.NewTensor(ort.NewShape(1, 2), []float32{float32(resizedImage.Rows()), float32(resizedImage.Cols())})
	if err != nil {
		return nil, fmt.Errorf("imageTensor input tensor error", err.Error())
	}
	defer imageTensor.Destroy()

	// 2.
	dataTensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(resizedImage.Rows()), int64(resizedImage.Cols())), imageCHW)
	if err != nil {
		return nil, fmt.Errorf("dataTensor input tensor error", err.Error())
	}
	defer dataTensor.Destroy()
	//]float32{0.17821342, 0.25188917}
	scaleFactorTensor, err := ort.NewTensor(ort.NewShape(1, 2), scale)
	if err != nil {
		return nil, fmt.Errorf("dataTensor input tensor error", err.Error())
	}
	defer scaleFactorTensor.Destroy()

	output0Tensor, err := ort.NewEmptyTensor[float32](ort.NewShape(300, 6))

	if err != nil {
		return nil, fmt.Errorf("output0Tensor output tensor error", err.Error())
	}

	defer output0Tensor.Destroy()

	output1Tensor, err := ort.NewEmptyTensor[int32](ort.NewShape(1))

	if err != nil {
		return nil, fmt.Errorf("output0Tensor output tensor error", err.Error())
	}

	defer output1Tensor.Destroy()
	// 检测（核心）
	err = docLayout.OnnxSession.Run([]ort.Value{imageTensor, dataTensor, scaleFactorTensor}, []ort.Value{output0Tensor, output1Tensor})
	if err != nil {
		fmt.Printf(err.Error())
	}

	return nil, err
}

// 缩放图片
func (docLayout *DocLayoutPlusLSession) resize(imageMat *gocv.Mat) (*gocv.Mat, []float32, error) {

	resizeMat := gocv.NewMat()

	err := gocv.Resize(*imageMat, &resizeMat, image.Pt(800, 800), 0, 0, gocv.InterpolationCubic)

	if err != nil {
		return nil, nil, fmt.Errorf("DocLayoutPlusLSession resize failed: %v", err)
	}

	scaleW := float32(800) / float32(imageMat.Cols())
	scaleH := float32(800) / float32(imageMat.Rows())

	return &resizeMat, []float32{scaleH, scaleW}, nil

}
