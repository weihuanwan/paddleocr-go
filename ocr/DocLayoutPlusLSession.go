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
	config      *PaddleOCRConfig
}

func (docLayout *DocLayoutPlusLSession) Run(originImage *gocv.Mat) ([]*DetResult, error) {
	// 缩放
	resizedImage, _, err := docLayout.resize(originImage)
	if err != nil {
		return nil, err
	}

	defer resizedImage.Close()
	// 归一化
	imageNormalize, err := common.Normalize(resizedImage, [3]float32{1, 2, 3}, [3]float32{1, 2, 3})

	if err != nil {
		return nil, err
	}

	defer imageNormalize.Close()
	//w := gocv.NewWindow("image")
	//w.ResizeWindow(imageNormalize.Cols(), imageNormalize.Rows())
	//w.IMShow(*imageNormalize)
	//w.WaitKey(0)
	// 转换 chw
	imageCHW := common.HWCToCHW(imageNormalize)

	// 描述
	shape := ort.NewShape(1, 3, int64(resizedImage.Rows()), int64(resizedImage.Cols()))
	// 创建张量
	detInputTensor, err := ort.NewTensor(shape, imageCHW)
	if err != nil {

		return nil, fmt.Errorf("create rec input tensor error", err.Error())
	}
	defer detInputTensor.Destroy()

	detOutputShape := ort.NewShape(1, 1, int64(resizedImage.Rows()), int64(resizedImage.Cols()))
	detOutputTensor, err := ort.NewEmptyTensor[float32](detOutputShape)

	if err != nil {
		return nil, fmt.Errorf("create det output tensor error", err.Error())
	}
	defer detOutputTensor.Destroy()
	// 检测（核心）
	err = docLayout.OnnxSession.Run([]ort.Value{detInputTensor}, []ort.Value{detOutputTensor})

	return nil, err
}

// 缩放图片
func (docLayout *DocLayoutPlusLSession) resize(imageMat *gocv.Mat) (*gocv.Mat, []int, error) {

	resizeMat := gocv.NewMat()
	err := gocv.Resize(*imageMat, &resizeMat, image.Pt(800, 800), 0, 0, gocv.InterpolationCubic)

	if err != nil {
		return nil, nil, fmt.Errorf("DocLayoutPlusLSession resize failed: %v", err)
	}
	return &resizeMat, make([]int, 0), nil

}
