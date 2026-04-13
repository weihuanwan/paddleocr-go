package common

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	"gocv.io/x/gocv"
)

func Round06(x float64) int {
	integer := math.Floor(x)
	frac := x - integer
	if frac >= 0.6 {
		return int(integer + 1)
	}
	return int(integer)
}

// Normalize 对图像进行归一化处理
// resizedImage: 输入图像（gocv.Mat）
// scale: 缩放系数
// mean/std: 均值和标准差，长度为3，分别对应 RGB
func Normalize(resizedImage *gocv.Mat, alpha, beta [3]float32) (*gocv.Mat, error) {
	c := resizedImage.Channels()
	gbrSplit := gocv.Split(*resizedImage)

	if c != 3 {
		return nil, fmt.Errorf("Normalize only supports 3-channel image, got %d", c)
	}

	// 对每个通道做归一化
	for i := 0; i < c; i++ {
		cpMat := gocv.NewMat()
		old := gbrSplit[i]
		err := old.ConvertTo(&cpMat, gocv.MatTypeCV32F)
		if err != nil {
			return nil, fmt.Errorf("Normalize convert to 32F error: %w", err)
		}
		old.Close()
		gbrSplit[i] = cpMat

		gbrSplit[i].MultiplyFloat(alpha[i])
		gbrSplit[i].AddFloat(beta[i])
	}

	result := gocv.NewMat()
	if err := gocv.Merge(gbrSplit, &result); err != nil {
		return nil, fmt.Errorf("Normalize merge error: %w", err)
	}

	return &result, nil
}

/*
*
转换chw
*/
func HWCToCHW(resizedImage *gocv.Mat) []float32 {
	//
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
func Resize(imageMat *gocv.Mat, resize [2]int) (*gocv.Mat, []float32, error) {
	resizeMat := gocv.NewMat()

	err := gocv.Resize(*imageMat, &resizeMat, image.Pt(resize[0], resize[1]), 0, 0, gocv.InterpolationCubic)

	if err != nil {
		return nil, nil, fmt.Errorf("wiredTableCells resize failed: %v", err)
	}

	scaleW := float32(resize[0]) / float32(imageMat.Cols())
	scaleH := float32(resize[1]) / float32(imageMat.Rows())

	return &resizeMat, []float32{scaleH, scaleW}, nil
}

func GetNormalizeAlphaBeta() ([3]float32, [3]float32) {
	scale := float32(1.0 / 255.0)
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	var alpha [3]float32
	var beta [3]float32

	for i := 0; i < 3; i++ {
		alpha[i] = scale / std[i]
		beta[i] = -mean[i] / std[i]
	}

	return alpha, beta
}

/*
解决同一个区域出现多个标签问题，取最高的，过滤最低的
*/
func NMSLayout(boxes []LayoutDetBox, iouSame, iouDiff float64) []LayoutDetBox {

	if len(boxes) == 0 {
		return boxes
	}

	// 对应 从大到小 排序
	sort.Slice(boxes, func(i, j int) bool {
		return boxes[i].Score > boxes[j].Score
	})

	var selected []LayoutDetBox

	// 对应 Python: while len(indices) > 0
	for len(boxes) > 0 {

		// current = indices[0]
		currentBox := boxes[0]
		// 当前的添加进去
		selected = append(selected, currentBox)

		var remaining []LayoutDetBox

		// for i in indices:
		for i := 1; i < len(boxes); i++ {
			// 获取下一个
			nextBox := boxes[i]

			// box_class
			nextBoxClass := nextBox.ClsId
			currentClass := currentBox.ClsId

			// iou
			iouValue := IoU(currentBox, nextBox)

			// threshold = iou_same if same class else iou_diff
			threshold := iouDiff
			// 判断类型是否一致
			if currentClass == nextBoxClass {
				// 如果类型是一致 使用0.6
				threshold = iouSame
			}

			// if iou < threshold → keep
			if iouValue < threshold {
				remaining = append(remaining, nextBox)
			}
		}

		boxes = remaining
	}

	return selected
}

func IoU(a, b LayoutDetBox) float64 {

	/**
	框 A                框 B
	(x1,y1)             (x1p,y1p)
	   ┌────────┐
	   │    A   │
	   │    ┌────────┐
	   │    │   B    │
	   └────┴────────┘
			(x2,y2)   (x2p,y2p)

	┌──────────────┐
	│              │
	│      A       │
	│      ┌──────────────┐
	│      │重叠区域 |     │
	└──────┴──────────────┘
	       │       B      │
	       └──────────────┘
	*/
	//minX := box.Point[0].X
	//minY := box.Point[0].Y
	//maxX := box.Point[1].X
	//maxY := box.Point[1].Y
	//
	//boxW, boxH := maxX-minX, maxY-minY
	//
	//// 默认矩形（四个顶点，顺序：左上、右上、右下、左下）
	//rect := []image.Point{
	//	{minX, minY},
	//	{maxX, minY},
	//	{maxX, maxY},
	//	{minX, maxY},
	//}

	//取坐标

	x1 := float64(a.Point[0])
	y1 := float64(a.Point[1])
	x2 := float64(a.Point[2])
	y2 := float64(a.Point[3])

	x1p := float64(b.Point[0])
	y1p := float64(b.Point[1])
	x2p := float64(b.Point[2])
	y2p := float64(b.Point[3])

	// intersection
	x1i := math.Max(x1, x1p)
	y1i := math.Max(y1, y1p)

	x2i := math.Min(x2, x2p)
	y2i := math.Min(y2, y2p)

	// 计算交集面积
	interW := math.Max(0, x2i-x1i+1)
	interH := math.Max(0, y2i-y1i+1)
	interArea := interW * interH
	// 计算两个框各自面积
	area1 := (x2 - x1 + 1) * (y2 - y1 + 1)
	area2 := (x2p - x1p + 1) * (y2p - y1p + 1)

	// 并集面积
	union := area1 + area2 - interArea
	if union <= 0 {
		return 0
	}
	// 交集面积 / 并集面积

	return interArea / union
}
func CropByBoxes(layoutDet *LayoutDetResult, imageMat gocv.Mat) (*gocv.Mat, error) {
	// 参数校验
	if layoutDet == nil {
		return nil, errors.New("layoutDet is nil")
	}
	if imageMat.Empty() {
		return nil, errors.New("imageMat is empty")
	}
	if len(layoutDet.Point) != 4 {
		return nil, errors.New("invalid point format, expected [xmin, ymin, xmax, ymax]")
	}

	xmin, ymin, xmax, ymax := layoutDet.Point[0], layoutDet.Point[1], layoutDet.Point[2], layoutDet.Point[3]

	rect := image.Rect(xmin, ymin, xmax, ymax)
	region := imageMat.Region(rect)

	// 无多边形时直接返回副本（避免原图释放后region失效）
	if len(layoutDet.PolygonPoints) == 0 {
		result := region.Clone()
		return &result, nil
	}

	// 创建mask
	mask := gocv.NewMatWithSize(region.Rows(), region.Cols(), gocv.MatTypeCV8U)
	defer mask.Close()

	// 构建局部坐标的多边形
	pts := make([]image.Point, 0, len(layoutDet.PolygonPoints))
	for _, p := range layoutDet.PolygonPoints {
		// 转换为相对于裁剪区域的坐标
		localX := p.X - xmin
		localY := p.Y - ymin
		// 检查转换后的坐标是否在有效范围内
		pts = append(pts, image.Pt(localX, localY))
	}

	// 填充多边形
	pointsVector := gocv.NewPointsVectorFromPoints([][]image.Point{pts})
	defer pointsVector.Close() // 修复：释放资源

	err := gocv.FillPoly(&mask, pointsVector, color.RGBA{255, 255, 255, 0})
	if err != nil {
		return nil, fmt.Errorf("failed to fill polygon points: %w", err)
	}
	// 创建结果图（透明背景更通用，或根据需求改为白色）
	result := gocv.NewMatWithSize(region.Rows(), region.Cols(), region.Type())

	// 设置背景色（白色）
	err = gocv.Rectangle(&result, image.Rect(0, 0, result.Cols(), result.Rows()),
		color.RGBA{255, 255, 255, 0}, -1)
	if err != nil {
		return nil, fmt.Errorf("failed to fill rectangle: %v", err)
	}
	// 应用mask拷贝有效区域
	err = region.CopyToWithMask(&result, mask)
	if err != nil {
		return nil, fmt.Errorf("failed to crop image: %v", err)
	}
	return &result, nil
}
