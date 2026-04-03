package table

import (
	"fmt"
	"image"
	"math"

	"github.com/weihuanwan/paddleocr-go/common"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

/*
*表格分类模型
 */
type TableClsOnnxSession struct {
	OnnxSession     *ort.DynamicAdvancedSession
	Labels          []string // 标签字典
	targetShortEdge int      // 标签字典
	CropSize        [2]int   //  (可选) 识方向模型裁剪大小 默认{224, 224}
	Alpha           [3]float32
	Beta            [3]float32
}

type TableClsResult struct {
	Label string  // 表格类型
	Score float32 // 置信度
}

func NewTableClsOnnxSession(onnxSession *ort.DynamicAdvancedSession) *TableClsOnnxSession {
	scale := float32(1.0 / 255.0)
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	var alpha [3]float32
	var beta [3]float32

	for i := 0; i < 3; i++ {
		alpha[i] = scale / std[i]
		beta[i] = -mean[i] / std[i]
	}
	//标签 :有线表格、无线表格
	var labels = []string{"wired_table", "wireless_table"}

	return &TableClsOnnxSession{
		onnxSession,
		labels,
		256,
		[2]int{224, 224},
		alpha,
		beta,
	}
}

func (tableCls *TableClsOnnxSession) Run(image *gocv.Mat) (*TableClsResult, error) {

	resizeImage, err := tableCls.resize(image)

	if err != nil {
		return nil, err
	}

	defer resizeImage.Close()

	cropImage := tableCls.crop(resizeImage)
	defer cropImage.Close()

	normalizeImage, err := common.Normalize(cropImage, tableCls.Alpha, tableCls.Beta)
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

		return nil, fmt.Errorf("TableClsOnnxSession create input tensor error %w", err)
	}
	defer clsInputTensor.Destroy()

	clsOutputShape := ort.NewShape(1, 2)
	clsOutputTensor, err := ort.NewEmptyTensor[float32](clsOutputShape)

	if err != nil {
		return nil, fmt.Errorf("TableClsOnnxSession create output tensor error %w", err)
	}
	defer clsOutputTensor.Destroy()
	// 检测（核心）
	err = tableCls.OnnxSession.Run([]ort.Value{clsInputTensor}, []ort.Value{clsOutputTensor})

	if err != nil {
		return nil, fmt.Errorf("TableClsOnnxSession run  error  %w", err)
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
	label := tableCls.Labels[maxIdx]

	return &TableClsResult{Label: label, Score: maxScore}, nil
}

// 缩放图片
func (tableCls *TableClsOnnxSession) resize(imageMat *gocv.Mat) (*gocv.Mat, error) {

	// 1. 设置 rgb

	rgb := gocv.NewMat()
	err := gocv.CvtColor(*imageMat, &rgb, gocv.ColorBGRToRGB)

	if err != nil {
		return nil, fmt.Errorf("colorBGRToRGB  failed: %w", err)
	}

	// 2.
	h := imageMat.Rows()
	w := imageMat.Rows()
	scale := float64(tableCls.targetShortEdge) / float64(min(h, w))

	resizeH := int(math.Round(float64(h) * scale))
	resizeW := int(math.Round(float64(w) * scale))

	resizeMat := gocv.NewMat()

	err = gocv.Resize(*imageMat, &resizeMat, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)

	if err != nil {
		return nil, fmt.Errorf("tableCls resize failed: %v", err)
	}

	return &resizeMat, nil

}

func (tableCls *TableClsOnnxSession) crop(resizeImage *gocv.Mat) *gocv.Mat {

	h := resizeImage.Rows()
	w := resizeImage.Cols()

	cw, ch := tableCls.CropSize[0], tableCls.CropSize[1]

	x1 := max(0, (w-cw)/2)
	y1 := max(0, (h-ch)/2)

	x2 := min(w, x1+cw)
	y2 := min(h, y1+ch)

	croppedRegion := resizeImage.Region(image.Rect(x1, y1, x2, y2))
	return &croppedRegion
}
