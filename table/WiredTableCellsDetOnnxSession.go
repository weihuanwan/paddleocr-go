package table

import (
	"fmt"

	"github.com/weihuanwan/paddleocr-go/common"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

/*
*
有线表格单元格检测模型
*/
type WiredTableCellsDetOnnxSession struct {
	OnnxSession *ort.DynamicAdvancedSession
	Alpha       [3]float32
	Beta        [3]float32

	Resize    [2]int  // 缩放大小，默认640*640
	Threshold float32 // 置信度 默认 0.3
}

func NewWiredTableCellsDetOnnxSession(onnxSession *ort.DynamicAdvancedSession) *WiredTableCellsDetOnnxSession {
	scale := float32(1.0 / 255.0)
	mean := []float32{0, 0, 0}
	std := []float32{1, 1, 1}
	var alpha [3]float32
	var beta [3]float32
	for i := 0; i < 3; i++ {
		alpha[i] = scale / std[i]
		beta[i] = -mean[i] / std[i]
	}
	return &WiredTableCellsDetOnnxSession{
		onnxSession,
		alpha,
		beta,
		[2]int{800, 800},
		0.3,
	}
}

func (wiredTableCells *WiredTableCellsDetOnnxSession) Run(originImage *gocv.Mat) ([]*TableCellsDetResult, error) {
	// 缩放
	resizedImage, scale, err := common.Resize(originImage, wiredTableCells.Resize)
	if err != nil {
		return nil, err
	}

	defer resizedImage.Close()

	imageNormalize, err := common.Normalize(resizedImage, wiredTableCells.Alpha, wiredTableCells.Beta)

	if err != nil {
		return nil, err
	}

	defer imageNormalize.Close()

	// 转换 chw
	imageCHW := common.HWCToCHW(imageNormalize)

	// 1.输入图像尺寸
	imageTensor, err := ort.NewTensor(ort.NewShape(1, 2), []float32{float32(resizedImage.Rows()), float32(resizedImage.Cols())})
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession imageTensor input tensor error %w", err.Error())
	}
	defer imageTensor.Destroy()

	// 2. 图像数据
	dataTensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(resizedImage.Rows()), int64(resizedImage.Cols())), imageCHW)
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession dataTensor input tensor error %w", err.Error())
	}

	defer dataTensor.Destroy()
	// 3. resize 缩放比例
	scaleFactorTensor, err := ort.NewTensor(ort.NewShape(1, 2), scale)
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession scaleFactorTensor input tensor error %w", err.Error())
	}
	defer scaleFactorTensor.Destroy()

	maxDet := int64(300)
	// 4.输出 最多300个检测框数量，每一个7个值 [ label_index, score, xmin, ymin, xmax, ymax,扩展参数]
	output0Tensor, err := ort.NewEmptyTensor[float32](ort.NewShape(maxDet, 7))

	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession output0Tensor output tensor error %w", err.Error())
	}

	defer output0Tensor.Destroy()

	// 5.输出实际框数量
	output1Tensor, err := ort.NewEmptyTensor[int32](ort.NewShape(1))
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession output1Tensor output tensor error %w", err.Error())
	}

	defer output1Tensor.Destroy()

	// 6. 像素级掩码,	最多 300 个检测框,每个框对应一个 200×200 的二值图
	output2Tensor, err := ort.NewEmptyTensor[int32](ort.NewShape(maxDet, 200, 200))
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession output2Tensor output tensor error %w", err.Error())
	}
	defer output2Tensor.Destroy()

	// 检测（核心）
	err = wiredTableCells.OnnxSession.Run([]ort.Value{imageTensor, dataTensor, scaleFactorTensor}, []ort.Value{output0Tensor, output1Tensor, output2Tensor})
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession OnnxSession.Run() error %w", err.Error())
	}

	return nil, err
}
