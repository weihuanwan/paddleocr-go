package ocr

import (
	"fmt"
	"image"
	"log"
	"math"
	"slices"
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

	Resize               [2]int   // 缩放
	Labels               []string // 标签字典
	LayoutMergeBoxesMode []string // 标签字典
	Threshold            float32  // 置信度
}
type LayoutDetBox struct {
	ClsId int
	Label string
	Score float32
	Order int

	Point [4]int

	Mask []int32
}

// 版面分析返回结果
type LayoutDetResult struct {
	ClsId         int           // 标签的 id
	Label         string        // 标签
	Score         float32       // 置信度
	Order         int           // 排序
	Point         []image.Point // 四边形 位置
	PolygonPoints []image.Point // 多边形位置
	Mask          []int32
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

	//标签
	var labels = []string{"abstract", "algorithm", "aside_text", "chart",
		"content", "display_formula", "doc_title", "figure_title", "footer",
		"footer_image", "footnote", "formula_number", "header", "header_image",
		"image", "inline_formula", "number", "paragraph_title",
		"reference", "reference_content", "seal", "table",
		"text", "vertical_text", "vision_footnote"}

	var layoutMergeBoxesMode = []string{"union", "union", "union", "large", "union",
		"large", "large", "union", "union", "union", "union", "union", "union", "union", "union", "large", "union",
		"large", "union", "union", "union", "union", "union", "union", "union"}

	return &LayoutDetSession{
		onnxSession,
		alpha,
		beta,
		[2]int{800, 800},
		labels,
		layoutMergeBoxesMode,
		0.3,
	}
}

func (layoutDet *LayoutDetSession) Run(originImage *gocv.Mat) ([]*LayoutDetResult, error) {
	// 缩放
	resizedImage, scale, err := layoutDet.resize(originImage)
	if err != nil {
		return nil, err
	}

	defer resizedImage.Close()
	// 归一化  [0.00392156862745098, 0.00392156862745098, 0.00392156862745098]
	// [-0.0, -0.0, -0.0]
	imageNormalize, err := common.Normalize(resizedImage, layoutDet.Alpha, layoutDet.Beta)

	if err != nil {
		return nil, err
	}

	defer imageNormalize.Close()

	// 转换 chw
	imageCHW := common.HWCToCHW(imageNormalize)

	// 1.输入图像尺寸
	imageTensor, err := ort.NewTensor(ort.NewShape(1, 2), []float32{float32(resizedImage.Rows()), float32(resizedImage.Cols())})
	if err != nil {
		return nil, fmt.Errorf("imageTensor input tensor error %w", err.Error())
	}
	defer imageTensor.Destroy()

	// 2. 图像数据
	dataTensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(resizedImage.Rows()), int64(resizedImage.Cols())), imageCHW)
	if err != nil {
		return nil, fmt.Errorf("dataTensor input tensor error %w", err.Error())
	}

	defer dataTensor.Destroy()
	// 3. resize 缩放比例
	scaleFactorTensor, err := ort.NewTensor(ort.NewShape(1, 2), scale)
	if err != nil {
		return nil, fmt.Errorf("scaleFactorTensor input tensor error %w", err.Error())
	}
	defer scaleFactorTensor.Destroy()

	maxDet := int64(300)
	// 4.输出 最多300个检测框数量，每一个7个值 [ label_index, score, xmin, ymin, xmax, ymax,扩展参数]
	output0Tensor, err := ort.NewEmptyTensor[float32](ort.NewShape(maxDet, 7))

	if err != nil {
		return nil, fmt.Errorf("output0Tensor output tensor error %w", err.Error())
	}

	defer output0Tensor.Destroy()

	// 5.输出实际框数量
	output1Tensor, err := ort.NewEmptyTensor[int32](ort.NewShape(1))
	if err != nil {
		return nil, fmt.Errorf("output1Tensor output tensor error %w", err.Error())
	}

	defer output1Tensor.Destroy()

	// 6. 像素级掩码,	最多 300 个检测框,每个框对应一个 200×200 的二值图
	output2Tensor, err := ort.NewEmptyTensor[int32](ort.NewShape(maxDet, 200, 200))
	if err != nil {
		return nil, fmt.Errorf("output2Tensor output tensor error %w", err.Error())
	}
	defer output2Tensor.Destroy()

	// 检测（核心）
	err = layoutDet.OnnxSession.Run([]ort.Value{imageTensor, dataTensor, scaleFactorTensor}, []ort.Value{output0Tensor, output1Tensor, output2Tensor})
	if err != nil {
		return nil, fmt.Errorf("layoutDet.OnnxSession.Run() error %w", err.Error())
	}

	// 处理结果
	layoutDetResult, err := layoutDet.formatOutput(
		output0Tensor.GetData(),
		output1Tensor.GetData(),
		output2Tensor.GetData(),
		originImage.Rows(),
		originImage.Cols(),
		scale)

	return layoutDetResult, err
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

func (layoutDet *LayoutDetSession) formatOutput(boxes []float32, count []int32,
	masks []int32, originImageH int, originImageW int, scale []float32) ([]*LayoutDetResult, error) {

	step := 7
	maskSize := 200 * 200
	layoutDetBoxs := make([]LayoutDetBox, 0)
	for i := 0; i < len(boxes); i += step {
		if boxes[i+0] > -1 && boxes[i+1] > layoutDet.Threshold {

			detIndex := i / step
			maskStart := detIndex * maskSize
			maskEnd := maskStart + maskSize
			// 获取像素级掩码
			mask := masks[maskStart:maskEnd]
			// 取这个位置的
			xmin := int(math.Round(float64(boxes[i+2])))
			ymin := int(math.Round(float64(boxes[i+3])))
			xmax := int(math.Round(float64(boxes[i+4])))
			ymax := int(math.Round(float64(boxes[i+5])))
			clsId := int(boxes[i])
			layoutDetResult := LayoutDetBox{
				ClsId: clsId,
				Score: boxes[i+1],
				Order: int(boxes[i+6]),
				Label: layoutDet.Labels[clsId],
				Point: [4]int{xmin, ymin, xmax, ymax},
				Mask:  mask,
			}
			layoutDetBoxs = append(layoutDetBoxs, layoutDetResult)
		}
	}

	// 解决同一个区域出现多个标签问题，取最高的，过滤最低的
	layoutDetResultNMS := NMSLayout(layoutDetBoxs, 0.6, 0.98)

	filteredBoxes := make([]LayoutDetBox, 0)
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

				// 过滤超大图的
				if boxArea <= int(areaThres*float64(imgArea)) {
					filteredBoxes = append(filteredBoxes, layoutDetResult)
				} else {
					log.Printf(
						"[LayoutDet] filter large image box, area=%d, imgArea=%d, ratio=%.2f",
						boxArea,
						imgArea,
						float64(boxArea)/float64(imgArea),
					)
				}
			} else {
				filteredBoxes = append(filteredBoxes, layoutDetResult)
			}
		}
	}

	// 解决一个大区域内存在小标签问题。如:一个表格内是存在文本标签问题
	filteredBoxesLen := len(filteredBoxes)
	keepMaskBoxes := make([]LayoutDetBox, 0, filteredBoxesLen)
	if filteredBoxesLen > 0 {
		// 225 2432 2913 3956  keep_mask = np.ones(len(boxes), dtype=bool)
		keepMask := slices.Repeat([]bool{true}, filteredBoxesLen)

		for categoryIndex := 0; categoryIndex < len(layoutDet.LayoutMergeBoxesMode); categoryIndex++ {
			mode := layoutDet.LayoutMergeBoxesMode[categoryIndex]
			if mode == "union" {
				continue
			}

			if mode == "large" {
				_, containedByOther := checkContainment(filteredBoxes, categoryIndex, mode)

				for i := 0; i < len(containedByOther); i++ {
					// 是true 的都是true
					keepMask[i] = keepMask[i] && !containedByOther[i]
				}
			} else if mode == "small" {
				// TODO python 源码不走这个分支 就不写这个了。

			}
		}
		// 过滤掉小框
		for i := 0; i < filteredBoxesLen; i++ {
			if keepMask[i] {
				keepMaskBoxes = append(keepMaskBoxes, filteredBoxes[i])
			}
		}
	}

	// 排序
	sort.Slice(keepMaskBoxes, func(i, j int) bool {
		return keepMaskBoxes[i].Order < keepMaskBoxes[j].Order
	})
	// 处理像素掩码得到一个多边形点位置
	polygonPoints := extractPolygonPointsByMasks(keepMaskBoxes, scale, "auto")

	layoutUnclipRatio := []float64{1.0, 1.0}
	unclipResult := unclipBoxes(keepMaskBoxes, layoutUnclipRatio)

	layoutDetResults := restructuredBoxes(
		unclipResult,
		polygonPoints,
		originImageH,
		originImageW,
	)
	return layoutDetResults, nil
}

func restructuredBoxes(results []*LayoutDetResult, polygonPoints [][]image.Point, originImageH int, originImageW int) []*LayoutDetResult {
	layoutDetResults := make([]*LayoutDetResult, 0, len(results))
	for i := 0; i < len(results); i++ {
		res := results[i]

		xmin, ymin, xmax, ymax := res.Point[0].X, res.Point[0].Y, res.Point[1].X, res.Point[1].Y

		xmin = max(0, xmin)
		ymin = max(0, ymin)
		xmax = min(originImageW, xmax)
		ymax = min(originImageH, ymax)
		if xmax <= xmin || ymax <= ymin {
			continue
		}
		if polygonPoints != nil && len(polygonPoints) > 0 {
			polygonPoint := polygonPoints[i]
			if polygonPoint != nil {
				res.PolygonPoints = polygonPoint
			}
		}

		res.Order = i + 1

		layoutDetResults = append(layoutDetResults, res)

	}

	return layoutDetResults
}

func unclipBoxes(boxes []LayoutDetBox, layoutUnclipRatio []float64) []*LayoutDetResult {

	layoutDetResults := make([]*LayoutDetResult, 0, len(boxes))

	for i := 0; i < len(boxes); i++ {
		box := boxes[i]
		width := box.Point[2] - box.Point[0]
		height := box.Point[3] - box.Point[1]

		newW := int(float64(width) * layoutUnclipRatio[0])
		newH := int(float64(height) * layoutUnclipRatio[1])

		centerX := box.Point[0] + width/2
		centerY := box.Point[1] + height/2

		newX1 := centerX - newW/2
		newY1 := centerY - newH/2
		newX2 := centerX + newW/2
		newY2 := centerY + newH/2

		layoutDetResults = append(layoutDetResults, &LayoutDetResult{
			ClsId: box.ClsId,
			Label: box.Label,
			Score: box.Score,
			Order: box.Order,
			Point: []image.Point{
				{newX1, newY1},
				{newX2, newY2},
			},
			PolygonPoints: nil,
			Mask:          nil,
		})

	}
	return layoutDetResults
}

// 处理重
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
				threshold = iouSame
			}

			// if iou < threshold → keep
			if iouValue < threshold {
				remaining = append(remaining, nextBox)
			} else {
				// 证明两个框的相似，那么就过滤掉最低的
				log.Println("iouValue < threshold")
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

func extractPolygonPointsByMasks(layoutDetBox []LayoutDetBox, scale []float32, layoutShapeMode string) [][]image.Point {

	scaleH := scale[0] / 4
	scaleW := scale[1] / 4
	hm := 200
	wm := 200
	// 找到最大的宽度
	maxBoxW := 0
	for i := 0; i < len(layoutDetBox); i++ {
		box := layoutDetBox[i]
		maxW := box.Point[3] - box.Point[0]
		if maxBoxW < maxW {
			maxBoxW = maxW
		}
	}

	polygonPoints := make([][]image.Point, 0)

	for i := 0; i < len(layoutDetBox); i++ {
		box := layoutDetBox[i]
		//  2501,214,107,2846
		minX := box.Point[0]
		minY := box.Point[1]
		maxX := box.Point[2]
		maxY := box.Point[3]

		boxW, boxH := maxX-minX, maxY-minY

		// 默认矩形（四个顶点，顺序：左上、右上、右下、左下）
		rect := []image.Point{
			{minX, minY},
			{maxX, minY},
			{maxX, maxY},
			{minX, maxY},
		}

		if boxW <= 0 || boxH <= 0 {
			polygonPoints = append(polygonPoints, rect)
			continue
		}
		// 14
		minW := int(math.Min(math.Max(0, math.Round(float64(minX)*float64(scaleW))), float64(wm)))
		maxW := int(math.Min(math.Max(0, math.Round(float64(maxX)*float64(scaleW))), float64(wm))) //183

		minH := int(math.Min(math.Max(0, math.Round(float64(minY)*float64(scaleH))), float64(hm))) //108
		maxH := int(math.Min(math.Max(0, math.Round(float64(maxY)*float64(scaleH))), float64(hm))) //176

		mask := box.Mask
		rows := maxH - minH + 1
		cols := (wm + maxW) - (wm + minW)
		mat := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8U)
		var count = int32(0)
		for row := minH; row <= maxH; row++ {
			maskStart := row*wm + minW
			maskEnd := row*wm + maxW
			// 找到
			m := mask[maskStart:maskEnd]
			for w := 0; w < len(m); w++ {
				h := row - minH
				mat.SetUCharAt(h, w, uint8(m[w]))
				count += m[w]
			}
		}
		if count == 0 {
			polygonPoints = append(polygonPoints, rect)
			mat.Close()
			continue
		}
		resizedMask := gocv.NewMat()
		err := gocv.Resize(mat, &resizedMask, image.Pt(boxW, boxH), 0, 0, gocv.InterpolationNearestNeighbor)

		if err != nil {
			panic(err)
		}

		maxAllowedDist := maxBoxW
		if boxW > int(float32(maxBoxW)*0.6) {
			maxAllowedDist = boxW
		}
		// 转换多点边界框
		polygon := mask2polygon(resizedMask, maxAllowedDist)

		if len(polygon) < 4 {
			polygonPoints = append(polygonPoints, rect)
			continue
		}

		if len(polygon) > 0 {
			for j := 0; j < len(polygon); j++ {
				poly := polygon[j]
				poly.X = poly.X + minX
				poly.Y = poly.Y + minY
				polygon[j] = poly
			}
		}

		if layoutShapeMode == "poly" {
			// 返回完整多边形
			polygonPoints = append(polygonPoints, polygon)
		} else if layoutShapeMode == "quad" {
			// 多边形转换 四边形
			quad := convertPolygonToQuad(polygon)
			if quad != nil && len(quad) > 0 {
				polygonPoints = append(polygonPoints, quad)
			} else {
				polygonPoints = append(polygonPoints, rect)
			}
		} else if layoutShapeMode == "auto" {
			// 多边形转换 四边形
			quad := convertPolygonToQuad(polygon)

			if len(quad) > 0 {
				iouQuad := calculatePolygonOverlapRatio(
					rect,
					quad,
					"union",
				)
				if iouQuad >= 0.95 {
					quad = rect
				}

				iouQuad = calculatePolygonOverlapRatio(
					polygon, quad, "union",
				)

			}
			polygonPoints = append(polygonPoints, polygon)
		} else {
			panic("Unsupported layoutShapeMode")
		}

	}

	return nil
}

func calculatePolygonOverlapRatio(rect []image.Point, quad []image.Point, mode string) float64 {

	return 0
}

func mask2polygon(mask gocv.Mat, maxAllowedDist int) []image.Point {
	epsilonRatio := 0.004
	// 获取位置
	pv := gocv.FindContours(mask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	cnt := pv.At(0)

	epsilon := epsilonRatio * gocv.ArcLength(cnt, true)
	approxCnt := gocv.ApproxPolyDP(cnt, epsilon, true)

	// 提取 多点边界框 顶点
	return extractCustomVertices(approxCnt.ToPoints(), maxAllowedDist)

}

func checkContainment(boxes []LayoutDetBox, categoryIndex int, mode string) ([]bool, []bool) {
	n := len(boxes)
	//我包含了别人
	containsOther := make([]bool, n)
	// 我被别人包含
	containedByOther := make([]bool, n)

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {

			if i == j {
				continue
			}
			if categoryIndex != -1 && mode != "" {

				if mode == "large" && boxes[j].ClsId == categoryIndex {
					if IsContained(boxes[i], boxes[j]) {
						containedByOther[i] = true
						containsOther[j] = true
					}
				}

				if mode == "small" && boxes[i].ClsId == categoryIndex {
					if IsContained(boxes[i], boxes[j]) {
						containedByOther[i] = true
						containsOther[j] = true
					}
				}
			} else {
				if IsContained(boxes[i], boxes[j]) {
					containedByOther[i] = true
					containsOther[j] = true
				}
			}
		}
	}

	return containsOther, containedByOther
}

func IsContained(box1, box2 LayoutDetBox) bool {

	x1, y1, x2, y2 := box1.Point[0], box1.Point[1], box1.Point[2], box1.Point[3]
	x1_p, y1_p, x2_p, y2_p := box2.Point[0], box2.Point[1], box2.Point[2], box2.Point[3]

	box1Area := (x2 - x1) * (y2 - y1)

	xi1 := max(x1, x1_p)
	yi1 := max(y1, y1_p)
	xi2 := min(x2, x2_p)
	yi2 := min(y2, y2_p)

	inter_width := max(0, xi2-xi1)
	inter_height := max(0, yi2-yi1)
	intersect_area := float64(inter_width) * float64(inter_height)

	var iou = float64(0)

	if box1Area > 0 {
		iou = intersect_area / float64(box1Area)
	}
	return iou >= 0.9
}

type PointInfo struct {
	Index    int
	IsConvex bool
	Angle    float32
	v1       image.Point
	v2       image.Point
}

// 提取自定义顶点
func extractCustomVertices(points []image.Point, maxAllowedDist int) []image.Point {
	n := len(points)
	pointInfos := make([]PointInfo, 0, n)
	adjustedMaxDist := float64(maxAllowedDist) * 0.3

	sharpAngleThresh := 45

	concaveIndices := make([]int, 0, n)
	/**
	计算每个位置是否是 凸凹点，如下表示:
	*/
	for i := 0; i < n; i++ {
		prevIndex := (i - 1 + n) % n
		currIndex := i
		nextIndex := (i + 1) % n
		// 上一个点 、当前点、下一个点，形成一个角
		prevP, currP, nextP := points[prevIndex], points[currIndex], points[nextIndex]

		// 上一个点 -当前点，下一个点减去当前点
		v1 := image.Pt(
			prevP.X-currP.X,
			prevP.Y-currP.Y,
		)
		v2 := image.Pt(
			nextP.X-currP.X,
			nextP.Y-currP.Y,
		)

		// 计算是不是凸角
		isConvexPoint := isConvex(prevP, currP, nextP)

		// 计算角度
		angle := angleBetweenVectors(v1, v2)

		pointInfos = append(pointInfos, PointInfo{
			Index:    currIndex,
			IsConvex: isConvexPoint,
			Angle:    float32(angle),
			v1:       v1,
			v2:       v2,
		})

		if !isConvexPoint {
			// 是凹点的索引
			concaveIndices = append(concaveIndices, i)
		}
	}

	/**
		2. 筛选需要保留的凹点，如下表示:

		*                 *

	*								*

		*
							*
	*

			*             *


	最后得到如下：

			*                 *

	*								*



	*

			*             *



	*/
	preserveConcave := make([]int, 0)
	concaveIndicesLen := len(concaveIndices)
	if concaveIndicesLen > 1 {
		groups := make([]int, 0)
		currentGroup := []int{concaveIndices[0]}

		for i := 1; i < concaveIndicesLen; i++ {
			// 判断是否连续 (包含首尾衔接)
			if concaveIndices[i]-concaveIndices[i-1] == 1 ||
				// 处理首尾闭合的特殊连续性
				(concaveIndices[i-1] == n-1 && concaveIndices[i] == 0) {

				currentGroup = append(currentGroup, concaveIndices[i])
			} else {
				if len(currentGroup) >= 2 {
					groups = append(groups, currentGroup...)
				}
				currentGroup = []int{concaveIndices[i]}
			}
		}
		// 处理最后一组
		if len(currentGroup) >= 2 {
			groups = append(groups, currentGroup...)
		}

		// 判断首尾是否是否存在
		if concaveIndicesLen >= 2 && concaveIndices[0] == 0 && concaveIndices[concaveIndicesLen-1] == n-1 {

			if slices.Contains(groups, 0) && slices.Contains(groups, n-1) {
				preserveConcave = append(preserveConcave, groups...)
			}
		} else {
			preserveConcave = append(preserveConcave, groups...)
		}

	}

	// 3. 确定初步保留点索引
	keptPoints := make([]int, 0)
	for i, info := range pointInfos {
		if info.IsConvex || (slices.Contains(preserveConcave, i) && info.Angle >= 120) {
			keptPoints = append(keptPoints, i)
		}
	}
	// 4. 距离插值 (处理两点之间距离过大的情况)
	finalIndices := make([]int, 0)
	for i := 0; i < len(keptPoints); i++ {
		// 当前id
		currIdx := keptPoints[i]
		// 下一个id
		nextIdx := keptPoints[(i+1)%len(keptPoints)]

		finalIndices = append(finalIndices, currIdx)

		currP, nextP := points[currIdx], points[nextIdx]

		// 两个点的向量差
		dist := math.Sqrt(math.Pow(float64(currP.X-nextP.X), 2) + math.Pow(float64(currP.Y-nextP.Y), 2))

		// 计算两个点的欧几里得距离（直线距离）
		if dist > adjustedMaxDist {
			var intermediate []int

			// 找出 current_idx 和 next_idx 之间被删除的那些点
			if nextIdx > currIdx {
				for j := currIdx + 1; j < nextIdx; j++ {
					intermediate = append(intermediate, j)
				}
			} else {
				for j := currIdx + 1; j < n; j++ {
					intermediate = append(intermediate, j)
				}
				for j := 0; j < nextIdx; j++ {
					intermediate = append(intermediate, j)
				}
			}

			// 如果中间点很多，只取一部分（均匀取）
			if len(intermediate) > 0 {
				numNeeded := int(math.Ceil(dist/adjustedMaxDist)) - 1
				// 如果小于这个添加全部进去
				if len(intermediate) <= numNeeded {
					for _, idx := range intermediate {
						finalIndices = append(finalIndices, idx)
					}
				} else {
					// 取一部分
					step := float64(len(intermediate)) / float64(numNeeded)
					for k := 0; k < numNeeded; k++ {
						finalIndices = append(finalIndices, intermediate[int(float64(k)*step)])
					}
				}
			}
		}
	}

	// 排序
	sort.Ints(finalIndices)
	res := make([]image.Point, 0, len(finalIndices))
	for _, idx := range finalIndices {
		info := pointInfos[idx]
		currP := points[idx]

		if info.IsConvex && math.Abs(float64(info.Angle-float32(sharpAngleThresh))) < 1 {
			log.Println("联系开发人员做出来")
			res = append(res, currP)
		} else {
			res = append(res, currP)
		}

	}

	return res
}

// 向量长度
func norm(p image.Point) float64 {
	return math.Sqrt(float64(p.X*p.X + p.Y*p.Y))
}

// 点积
func dot(p1, p2 image.Point) float64 {
	return float64(p1.X*p2.X + p1.Y*p2.Y)
}

// 两个向量夹角（角度）
func angleBetweenVectors(v1, v2 image.Point) float64 {
	n1 := norm(v1)
	n2 := norm(v2)

	if n1 == 0 || n2 == 0 {
		return 0
	}

	cosTheta := dot(v1, v2) / (n1 * n2)

	// 防止浮点误差
	if cosTheta > 1 {
		cosTheta = 1
	}
	if cosTheta < -1 {
		cosTheta = -1
	}

	angleRad := math.Acos(cosTheta)
	return angleRad * 180 / math.Pi
}

func isConvex(prev image.Point, curr image.Point, next image.Point) bool {
	// 1. 计算进来的向量 (从 上一个点 指向 当前点)
	v1 := image.Pt(
		curr.X-prev.X,
		curr.Y-prev.Y,
	)
	// 2. 计算出去的向量 (从 当前点 指向 下一个点)
	v2 := image.Pt(
		next.X-curr.X,
		next.Y-curr.Y,
	)
	//# 3. 计算二维叉积 (Cross Product)
	//# 公式：x1*y2 - x2*y1
	cross := v1.X*v2.Y - v1.Y*v2.X
	return cross < 0
}
func convertPolygonToQuad(polygon []image.Point) []image.Point {

	if len(polygon) < 3 {
		return nil
	}

	pv := gocv.NewPointVectorFromPoints(polygon)
	defer pv.Close()
	minRect := gocv.MinAreaRect(pv)
	ptsMat := gocv.NewMat()
	defer ptsMat.Close()
	err := gocv.BoxPoints(minRect, &ptsMat)
	if err != nil {
		panic(err)
	}

	// 从 Mat 中提取四个点
	if ptsMat.Rows() != 4 || ptsMat.Cols() != 2 {
		return nil
	}

	// 读取浮点坐标
	floatPts := make([]gocv.Point2f, 4)
	for i := 0; i < 4; i++ {
		// 获取第 i 行，每行两个 float32
		x := ptsMat.GetFloatAt(i, 0)
		y := ptsMat.GetFloatAt(i, 1)
		// 转换为整数坐标（四舍五入）
		floatPts[i] = gocv.Point2f{
			X: x,
			Y: y,
		}
	}

	//  按角度排序（逆时针/顺时针），然后调整起点为左上角
	// 计算中心点
	var cx, cy float64
	for _, p := range floatPts {
		cx += float64(p.X)
		cy += float64(p.Y)
	}
	cx /= 4
	cy /= 4

	// 按角度排序（和 Python 一样）
	sort.Slice(floatPts, func(i, j int) bool {
		angleI := math.Atan2(float64(floatPts[i].Y)-cy, float64(floatPts[i].X)-cx)
		angleJ := math.Atan2(float64(floatPts[j].Y)-cy, float64(floatPts[j].X)-cx)
		return angleI < angleJ
	})

	// 找左上角（x+y 最小）
	topLeftIdx := 0
	minSum := floatPts[0].X + floatPts[0].Y
	for i := 1; i < 4; i++ {
		sum := floatPts[i].X + floatPts[i].Y
		if sum < minSum {
			minSum = sum
			topLeftIdx = i
		}
	}

	// 7. roll（np.roll 等价实现）
	result := make([]image.Point, 4)
	for i := 0; i < 4; i++ {
		p := floatPts[(topLeftIdx+i)%4]
		result[i] = image.Point{
			X: int(math.Round(float64(p.X))),
			Y: int(math.Round(float64(p.Y))),
		}
	}
	return result
}
