package table

import (
	"fmt"
	"math"

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
	Labels    []string
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
	var labels = []string{"cell"}
	return &WiredTableCellsDetOnnxSession{
		onnxSession,
		alpha,
		beta,
		[2]int{640, 640},
		0.3,
		labels,
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
	// 3. resize 缩放比例 [[1.6161616  0.89385474]]
	scaleFactorTensor, err := ort.NewTensor(ort.NewShape(1, 2), scale)
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession scaleFactorTensor input tensor error %w", err.Error())
	}
	defer scaleFactorTensor.Destroy()

	maxDet := int64(300)
	// 4.输出 最多300个检测框数量，每一个6个值 [ label_index, score, xmin, ymin, xmax, ymax]
	output0Tensor, err := ort.NewEmptyTensor[float32](ort.NewShape(maxDet, 6))

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

	// 检测（核心）
	err = wiredTableCells.OnnxSession.Run([]ort.Value{imageTensor, dataTensor, scaleFactorTensor}, []ort.Value{output0Tensor, output1Tensor})
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession OnnxSession.Run() error %w", err.Error())
	}

	return nil, err
}
func (wiredTableCells *WiredTableCellsDetOnnxSession) formatOutput(boxes []float32, count []int32,
	masks []int32, originImageH int, originImageW int,
	scale []float32) ([]*common.LayoutDetResult, error) {

	step := 7
	maskSize := 200 * 200
	layoutDetBoxs := make([]common.LayoutDetBox, 0)
	// 1. 处理图片
	for i := 0; i < len(boxes); i += step {

		if boxes[i+0] > -1 && boxes[i+1] > wiredTableCells.Threshold {
			detIndex := i / step
			maskStart := detIndex * maskSize
			maskEnd := maskStart + maskSize
			// 获取像素级掩码
			mask := masks[maskStart:maskEnd]
			clsId := int(boxes[i])
			score := boxes[i+1]
			// 取这个位置的
			xmin := int(math.Round(float64(boxes[i+2])))
			ymin := int(math.Round(float64(boxes[i+3])))
			xmax := int(math.Round(float64(boxes[i+4])))
			ymax := int(math.Round(float64(boxes[i+5])))

			layoutDetResult := common.LayoutDetBox{
				ClsId: clsId,
				Score: score,
				Label: wiredTableCells.Labels[clsId],
				Point: [4]int{xmin, ymin, xmax, ymax},
				Mask:  mask,
			}
			layoutDetBoxs = append(layoutDetBoxs, layoutDetResult)
		}
	}

	// 解决同一个区域出现多个标签问题，取最高的，过滤最低的
	layoutDetResultNMS := common.NMSLayout(layoutDetBoxs, 0.6, 0.98)

	filteredBoxes := make([]common.LayoutDetBox, 0)
	// 处理版面分析把当前输入的图片当做图片输出问题
	if len(layoutDetResultNMS) > 0 {
		areaThres := 0.93
		if originImageW > originImageH {
			areaThres = 0.82
		}
		imgArea := originImageH * originImageW

		for i := 0; i < len(layoutDetResultNMS); i++ {
			layoutDetResult := layoutDetResultNMS[i]
			// 判断是否是图片
			if layoutDetResult.Label == "image" {
				xmin := max(0, layoutDetResult.Point[0])
				ymin := max(0, layoutDetResult.Point[1])
				xmax := min(originImageW, layoutDetResult.Point[2])
				ymax := min(originImageH, layoutDetResult.Point[3])
				boxArea := (xmax - xmin) * (ymax - ymin)
				// 如果某个 image 框面积接近整张图面积，就把这个框过滤掉
				if boxArea <= int(areaThres*float64(imgArea)) {
					filteredBoxes = append(filteredBoxes, layoutDetResult)
				}
			} else {
				filteredBoxes = append(filteredBoxes, layoutDetResult)
			}
		}
	}

	// 结果组装
	layoutDetResults := restructuredBoxes(
		filteredBoxes,
		originImageH,
		originImageW,
	)
	return layoutDetResults, nil
}

func restructuredBoxes(boxes []common.LayoutDetBox, originImageH, originImageW int) []*common.LayoutDetResult {

	layoutDetResults := make([]*common.LayoutDetResult, 0, len(boxes))

	for i := 0; i < len(boxes); i++ {
		box := boxes[i]
		xmin := box.Point[0]
		ymin := box.Point[1]
		xmax := box.Point[2]
		ymax := box.Point[3]

		xmin = max(0, xmin)
		ymin = max(0, ymin)
		xmax = min(originImageW, xmax)
		ymax = min(originImageH, ymax)
		if xmax <= xmin || ymax <= ymin {
			continue
		}

		layoutDetResult := &common.LayoutDetResult{
			ClsId: box.ClsId,
			Label: box.Label,
			Score: box.Score,
			Order: box.Order,
			Point: []int{
				xmin, ymin, xmax, ymax,
			},
		}
		layoutDetResults = append(layoutDetResults, layoutDetResult)

	}
	return layoutDetResults
}
