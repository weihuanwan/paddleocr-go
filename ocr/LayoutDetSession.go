package ocr

import (
	"fmt"
	"image"
	"math"
	"sort"

	"github.com/weihuanwan/paddleocr-go/common"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

/*
* PP-DocLayoutV3 版面区域检测模块
 */
type LayoutDetSession struct {
	OnnxSession *ort.DynamicAdvancedSession
	Alpha       [3]float32
	Beta        [3]float32

	Resize [2]int   // 缩放
	Labels []string // 标签字典
}
type LayoutDetResult struct {
	ClsId int
	Label string
	Score float32
	Order int

	Point []*image.Point
}

func NewLayoutDetSession(onnxSession *ort.DynamicAdvancedSession) *LayoutDetSession {
	scale := float32(1.0 / 255.0)
	mean := []float32{0, 0, 0}
	std := []float32{1, 1, 1}
	var alpha [3]float32
	var beta [3]float32

	for i := 0; i < 3; i++ {
		alpha[i] = scale / std[i]
		beta[i] = -mean[i] / std[i]
	}

	var labels = []string{"abstract", "algorithm", "aside_text", "chart",
		"content", "display_formula", "doc_title", "figure_title", "footer",
		"footer_image", "footnote", "formula_number", "header", "header_image",
		"image", "inline_formula", "number", "paragraph_title",
		"reference", "reference_content", "seal", "table",
		"text", "vertical_text", "vision_footnote"}
	return &LayoutDetSession{
		onnxSession,
		alpha,
		beta,
		[2]int{800, 800},
		labels,
	}
}

func (layoutDet *LayoutDetSession) Run(originImage *gocv.Mat) ([]*DetResult, error) {
	// 缩放
	resizedImage, scale, err := layoutDet.resize(originImage)
	if err != nil {
		return nil, err
	}

	defer resizedImage.Close()
	// 归一化 # [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
	//# [-0.0, -0.0, -0.0]
	imageNormalize, err := common.Normalize(resizedImage, layoutDet.Alpha, layoutDet.Beta)

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

	scaleFactorTensor, err := ort.NewTensor(ort.NewShape(1, 2), scale)
	if err != nil {
		return nil, fmt.Errorf("dataTensor input tensor error", err.Error())
	}
	defer scaleFactorTensor.Destroy()

	// 最多输出 300个检测，每一个7个值 [ label_index, score, xmin, ymin, xmax, ymax,扩展参数]
	output0Tensor, err := ort.NewEmptyTensor[float32](ort.NewShape(300, 7))

	if err != nil {
		return nil, fmt.Errorf("output0Tensor output tensor error", err.Error())
	}

	defer output0Tensor.Destroy()

	output1Tensor, err := ort.NewEmptyTensor[int32](ort.NewShape(1))

	if err != nil {
		return nil, fmt.Errorf("output0Tensor output tensor error", err.Error())
	}

	defer output1Tensor.Destroy()

	output2Tensor, err := ort.NewEmptyTensor[int32](ort.NewShape(300, 200, 200))

	if err != nil {
		return nil, fmt.Errorf("output0Tensor output tensor error", err.Error())
	}

	defer output2Tensor.Destroy()

	// 检测（核心）
	err = layoutDet.OnnxSession.Run([]ort.Value{imageTensor, dataTensor, scaleFactorTensor}, []ort.Value{output0Tensor, output1Tensor, output2Tensor})
	if err != nil {
		fmt.Printf(err.Error())
	}

	layoutDet.formatOutput(output0Tensor.GetData(), output1Tensor.GetData(), output2Tensor.GetData())

	return nil, err
}

// 缩放图片
func (layoutDet *LayoutDetSession) resize(imageMat *gocv.Mat) (*gocv.Mat, []float32, error) {

	resizeMat := gocv.NewMat()

	err := gocv.Resize(*imageMat, &resizeMat, image.Pt(layoutDet.Resize[0], layoutDet.Resize[1]), 0, 0, gocv.InterpolationCubic)

	if err != nil {
		return nil, nil, fmt.Errorf("DocLayoutPlusLSession resize failed: %v", err)
	}

	scaleW := float32(layoutDet.Resize[0]) / float32(imageMat.Cols())
	scaleH := float32(layoutDet.Resize[1]) / float32(imageMat.Rows())

	return &resizeMat, []float32{scaleH, scaleW}, nil

}

func (layoutDet *LayoutDetSession) formatOutput(boxes []float32, i1 []int32, j1 []int32) {

	step := 7
	threshold := float32(0.3)

	layoutDetResults := make([]*LayoutDetResult, 0)
	for i := 0; i < len(boxes); i += step {
		//// 只处理坐标 x1 y1 x2 y2
		//boxes[i+2] = float32(int(math.Round(float64(boxes[i+2]))))
		//boxes[i+3] = float32(int(math.Round(float64(boxes[i+3]))))
		//boxes[i+4] = float32(int(math.Round(float64(boxes[i+4]))))
		//boxes[i+5] = float32(int(math.Round(float64(boxes[i+5]))))

		if boxes[i+0] > -1 && boxes[i+1] > threshold {
			xmin := boxes[i+2]
			ymin := boxes[i+3]
			xmax := boxes[i+4]
			ymax := boxes[i+5]
			minP := image.Point{int(math.Round(float64(xmin))), int(math.Round(float64(ymin)))}
			maxP := image.Point{int(math.Round(float64(xmax))), int(math.Round(float64(ymax)))}
			layoutDetResult := &LayoutDetResult{
				ClsId: int(boxes[i+0]),
				Score: boxes[i+1],
				Label: layoutDet.Labels[i],
				Point: []*image.Point{&minP, &maxP},
			}
			layoutDetResults = append(layoutDetResults, layoutDetResult)
		}
	}
	fmt.Println(layoutDetResults)
}

func NMSLayout(boxes []*LayoutDetResult, iouSame, iouDiff float64) []*LayoutDetResult {

	if len(boxes) == 0 {
		return boxes
	}

	// 对应 从大到小 排序
	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Score > boxes[j].Score
	})

	var selected []*LayoutDetResult

	// 对应 Python: while len(indices) > 0
	for len(boxes) > 0 {

		// current = indices[0]
		current := boxes[0]
		selected = append(selected, current)

		var remaining []*LayoutDetResult

		// for i in indices:
		for i := 1; i < len(boxes); i++ {
			box := boxes[i]

			// box_class
			boxClass := box.ClsId
			currentClass := current.ClsId

			// iou
			iouValue := IoU(current, box)

			// threshold = iou_same if same class else iou_diff
			threshold := iouDiff
			if currentClass == boxClass {
				threshold = iouSame
			}

			// if iou < threshold → keep
			if iouValue < threshold {
				remaining = append(remaining, box)
			}
		}

		boxes = remaining
	}

	return selected
}
func IoU(a, b *LayoutDetResult) float64 {
	x1 := float64(a.Point[0].X)
	y1 := float64(a.Point[0].Y)
	x2 := float64(a.Point[1].X)
	y2 := float64(a.Point[1].Y)

	x1p := float64(b.Point[0].X)
	y1p := float64(b.Point[0].Y)
	x2p := float64(b.Point[1].X)
	y2p := float64(b.Point[1].Y)

	// intersection
	x1i := math.Max(x1, x1p)
	y1i := math.Max(y1, y1p)
	x2i := math.Min(x2, x2p)
	y2i := math.Min(y2, y2p)

	interW := math.Max(0, x2i-x1i+1)
	interH := math.Max(0, y2i-y1i+1)

	interArea := interW * interH

	area1 := (x2 - x1 + 1) * (y2 - y1 + 1)
	area2 := (x2p - x1p + 1) * (y2p - y1p + 1)

	union := area1 + area2 - interArea
	if union <= 0 {
		return 0
	}

	return interArea / union
}
