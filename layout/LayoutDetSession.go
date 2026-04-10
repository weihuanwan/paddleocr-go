package layout

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"math"
	"slices"
	"sort"

	go_clipper2 "github.com/bolom009/go-clipper2"
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

	Resize               [2]int   // 缩放大小，默认800*800
	Labels               []string // 标签字典
	LayoutMergeBoxesMode []string // 标签字典
	Threshold            float32  // 置信度 默认 0.3
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
	Point         []int         // 四边形 4个点位置
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
	var labels = []string{
		"abstract", "algorithm", "aside_text", "chart", "content",
		"display_formula", "doc_title", "figure_title", "footer", "footer_image",
		"footnote", "formula_number", "header", "header_image", "image",
		"inline_formula", "number", "paragraph_title", "reference", "reference_content",
		"seal", "table", "text", "vertical_text", "vision_footnote"}
	/**
	和标签一一对应。如： table -> large 保留大框的
	在实际项目中识别出一个表格出来，但是里面还有文字也识别出来了，所以要过滤小文本框保留大表格框,除非表格还包含了图片这些
	*/
	var layoutMergeBoxesMode = []string{
		"union", "union", "union", "large", "union",
		"large", "large", "union", "union", "union",
		"union", "union", "union", "union", "union",
		"large", "union", "large", "union", "union",
		"union", "large", "union", "union", "union"}

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
	err = layoutDet.OnnxSession.Run([]ort.Value{imageTensor, dataTensor, scaleFactorTensor}, []ort.Value{output0Tensor, output1Tensor, output2Tensor})
	if err != nil {
		return nil, fmt.Errorf("LayoutDetSession OnnxSession.Run() error %w", err.Error())
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
		return nil, nil, fmt.Errorf("LayoutDetSession resize failed: %v", err)
	}

	scaleW := float32(layoutDet.Resize[0]) / float32(imageMat.Cols())
	scaleH := float32(layoutDet.Resize[1]) / float32(imageMat.Rows())

	return &resizeMat, []float32{scaleH, scaleW}, nil

}

func (layoutDet *LayoutDetSession) formatOutput(boxes []float32, count []int32,
	masks []int32, originImageH int, originImageW int,
	scale []float32) ([]*LayoutDetResult, error) {

	step := 7
	maskSize := 200 * 200
	layoutDetBoxs := make([]LayoutDetBox, 0)
	// 1. 处理图片
	for i := 0; i < len(boxes); i += step {

		if boxes[i+0] > -1 && boxes[i+1] > layoutDet.Threshold {
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
			order := int(boxes[i+6])
			layoutDetResult := LayoutDetBox{
				ClsId: clsId,
				Score: score,
				Order: order,
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

	// 解决一个大区域内存在小标签问题。如:一个表格内是存在文本标签问题
	filteredBoxesLen := len(filteredBoxes)
	keepMaskBoxes := make([]LayoutDetBox, 0, filteredBoxesLen)
	if filteredBoxesLen > 0 {
		keepMask := slices.Repeat([]bool{true}, filteredBoxesLen)

		for categoryIndex := 0; categoryIndex < len(layoutDet.LayoutMergeBoxesMode); categoryIndex++ {
			// 获取该标签合并方式
			mode := layoutDet.LayoutMergeBoxesMode[categoryIndex]
			if mode == "union" {
				continue
			}
			if mode == "large" { // 保留大框 排除小框
				_, containedByOther := checkContainment(filteredBoxes, categoryIndex, mode)
				for i := 0; i < len(containedByOther); i++ {
					// 是true 的都是true
					keepMask[i] = keepMask[i] && !containedByOther[i]
				}
			} else if mode == "small" { // 保留小框 排除大框
				containsOther, containsByOther := checkContainment(filteredBoxes, categoryIndex, mode)
				for i := 0; i < len(containsByOther); i++ {
					containByOther := !containsByOther[i]
					containOther := containsOther[i]
					condition := containByOther || containOther
					keepMask[i] = keepMask[i] && condition
				}

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
	/**
	处理像素掩码得到一个多边形点位置（重点核心地方）
	*/
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

		xmin, ymin, xmax, ymax := res.Point[0], res.Point[1], res.Point[2], res.Point[3]

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
			Point: []int{
				newX1, newY1, newX2, newY2,
			},
			PolygonPoints: nil,
			Mask:          nil,
		})

	}
	return layoutDetResults
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

func extractPolygonPointsByMasks(layoutDetBox []LayoutDetBox,
	scale []float32, layoutShapeMode string) [][]image.Point {

	scaleH := scale[0] / 4
	scaleW := scale[1] / 4

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
	// 默认是200
	hm := 200
	wm := 200
	for i := 0; i < len(layoutDetBox); i++ {
		box := layoutDetBox[i]
		// 原图片坐标系
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

		// 原图坐标 → mask坐标 (坐标系转换（大图 → 小图）)
		minW := int(math.Min(math.Max(0, float64(math.Round(float64(minX)*float64(scaleW)))), float64(wm)))
		maxW := int(math.Min(math.Max(0, float64(math.Round(float64(maxX)*float64(scaleW)))), float64(wm)))

		minH := int(math.Min(math.Max(0, float64(math.Round(float64(minY)*float64(scaleH)))), float64(hm)))
		maxH := int(math.Min(math.Max(0, float64(math.Round(float64(maxY)*float64(scaleH)))), float64(hm)))

		mask := box.Mask
		rows := maxH - minH
		cols := (wm + maxW) - (wm + minW)
		mat := gocv.NewMatWithSize(rows, cols, gocv.MatTypeCV8U)

		var count = int32(0)

		// 把模型输出的掩码处理
		for row := minH; row < maxH; row++ {
			h := row - minH
			maskStart := row*wm + minW
			maskEnd := row*wm + maxW
			// 找到这个位置的数据
			m := mask[maskStart:maskEnd]
			//fmt.Printf("%v", m)
			//fmt.Println()
			for w := 0; w < len(m); w++ {
				// 设置改坐标
				mat.SetUCharAt(h, w, uint8(m[w]))
				count += m[w]
			}
		}
		if count == 0 {
			polygonPoints = append(polygonPoints, rect)
			continue
		}
		//// 按 200 行，每行 200 个元素打印
		//for i := 0; i < 200; i++ {
		//	for j := 0; j < 200; j++ {
		//		idx := i*200 + j
		//		fmt.Printf("%v ", mask[idx]) // 元素之间用空格分隔
		//	}
		//	fmt.Println() // 每打印完一行换行
		//}
		resizedMask := gocv.NewMat()
		err := gocv.Resize(mat, &resizedMask, image.Pt(boxW, boxH), 0, 0, gocv.InterpolationNearestNeighbor)

		if err != nil {
			mat.Close()
			resizedMask.Close()
			panic("failed to resize image")
		}
		mat.Close()
		//resizedMaskW := gocv.NewWindow(`resizedMask`)
		//resizedMaskW.ResizeWindow(boxW, boxH)
		//resizedMaskW.IMShow(resizedMask)
		//resizedMaskW.WaitKey(0)

		maxAllowedDist := maxBoxW
		if boxW > int(float32(maxBoxW)*0.6) {
			maxAllowedDist = boxW
		}
		// 转换多点边界框（核心）
		polygon := mask2polygon(resizedMask, maxAllowedDist)
		resizedMask.Close()
		if len(polygon) < 4 {
			polygonPoints = append(polygonPoints, rect)
			continue
		}

		// 在检测时候是缩图的格式，现在把坐标转换原来的坐标
		if len(polygon) > 0 {
			for j := 0; j < len(polygon); j++ {
				poly := polygon[j]
				poly.X = poly.X + minX
				poly.Y = poly.Y + minY
				polygon[j] = poly
			}
		}

		if layoutShapeMode == "poly" {
			polygonPoints = append(polygonPoints, polygon)
			// 返回完整多边形
			continue
		} else if layoutShapeMode == "quad" {
			// 多边形转换 四边形
			quad := convertPolygonToQuad(polygon)
			if quad != nil && len(quad) > 0 {
				polygonPoints = append(polygonPoints, quad)

			} else {
				polygonPoints = append(polygonPoints, rect)
			}
			continue
		} else if layoutShapeMode == "auto" {
			iouThreshold := 0.8
			// 多边形转换 四边形
			quad := convertPolygonToQuad(polygon)
			if quad != nil && len(quad) > 0 {
				quadList := quad
				rectList := rect
				iouQuad1 := CalculatePolygonOverlapRatio(
					rectList,
					quadList,
					"union",
				)

				if iouQuad1 >= 0.95 {
					quad = rect
				}

				// 判断用四边形（quad）替代多边形（polygon），会不会“失真太多”。
				iouQuad2 := CalculatePolygonOverlapRatio(
					polygon, quadList, "union",
				)

				var prePoly []image.Point
				if len(polygonPoints) > 0 {
					prePoly = polygonPoints[len(polygonPoints)-1]
				}
				iouPre := float64(0)
				if prePoly != nil {
					iouPre = CalculatePolygonOverlapRatio(
						prePoly, rect, "small",
					)
				}
				if iouQuad2 >= iouThreshold && iouPre < 0.01 {
					polygonPoints = append(polygonPoints, quad)
					continue
				}
			}
			polygonPoints = append(polygonPoints, polygon)
		} else {
			panic("invalid layoutShapeMode")
		}

	}
	return polygonPoints
}
func CalculatePolygonOverlapRatio(
	polygon1, polygon2 []image.Point,
	mode string,
) float64 {
	// 参数校验（与 Python 类似）
	if len(polygon1) < 3 || len(polygon2) < 3 {
		return 0
	}

	// 转换为 clipper Paths64（相当于 Polygon()）
	subject := pointsToPaths64(polygon1)
	clip := pointsToPaths64(polygon2)

	// 使用 NonZero 填充规则（Shapely 默认行为）
	fillRule := go_clipper2.NonZero

	// 计算交集（相当于 poly1.intersection(poly2)）
	intersection := go_clipper2.IntersectWithClipPaths64(subject, clip, fillRule)

	// 交集面积（多个多边形求和，相当于 .area）
	intersectionArea := paths64TotalArea(intersection)

	// 计算并集（相当于 poly1.union(poly2)）
	union := go_clipper2.UnionWithClipPaths64(subject, clip, fillRule)

	// 并集面积（多个多边形求和，相当于 .area）
	unionArea := paths64TotalArea(union)

	// 原始多边形面积（用于 small/large 模式）
	area1 := paths64TotalArea(subject)
	area2 := paths64TotalArea(clip)

	// 根据 mode 计算比例（与 Python 完全一致）
	switch mode {
	case "union", "":
		if unionArea == 0 {
			return 0
		}
		return intersectionArea / unionArea

	case "small":
		smallArea := math.Min(area1, area2)
		if smallArea == 0 {
			return 0
		}
		return intersectionArea / smallArea

	case "large":
		largeArea := math.Max(area1, area2)
		if largeArea == 0 {
			return 0
		}
		return intersectionArea / largeArea

	default:
		panic("invalid layoutShapeMode")
	}
}

// pointsToPaths64 将 []image.Point 转换为 clipper.Paths64
func pointsToPaths64(points []image.Point) go_clipper2.Paths64 {
	path := make(go_clipper2.Path64, len(points))
	for i, p := range points {
		path[i] = go_clipper2.Point64{X: int64(p.X), Y: int64(p.Y)}
	}
	return go_clipper2.Paths64{path}
}

// paths64TotalArea 计算 Paths64 的总面积（遍历所有多边形求和）
func paths64TotalArea(paths go_clipper2.Paths64) float64 {
	total := 0.0
	for _, p := range paths {
		total += math.Abs(go_clipper2.Area64(p))
	}
	return total
}
func mask2polygon(mask gocv.Mat, maxAllowedDist int) []image.Point {
	epsilonRatio := 0.004
	// 获取位置
	contours := gocv.FindContours(mask, gocv.RetrievalExternal, gocv.ChainApproxSimple)
	// 1. 找最大轮廓（关键！！！）
	var maxCnt gocv.PointVector
	maxArea := 0.0

	for i := 0; i < contours.Size(); i++ {
		cnt := contours.At(i)
		area := gocv.ContourArea(cnt)

		if area > maxArea {
			maxArea = area
			maxCnt = cnt
		}
	}
	// 2. epsilon
	epsilon := epsilonRatio * gocv.ArcLength(maxCnt, true)
	//  3. 多边形拟合
	approxCnt := gocv.ApproxPolyDP(maxCnt, epsilon, true)

	points := approxCnt.ToPoints()
	// 提取 多点边界框 顶点 [[  0   0], [  0  46], [475  46], [475   0]]
	return extractCustomVertices(points, maxAllowedDist)

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
	2. 筛选需要保留的凹点
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
		// 如果是凹点 ，并且角度是大于 120)才保留
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

		// 处理凸点的
		/**
		移动前（尖刺状）              移动后（钝化）

			   ╱╲                            ╱  ╲
			  ╱  ╲                          ╱    ╲
			 ╱    ╲                        ╱      ╲
			╱      ╲                      ╱        ╲
		   ●────────●                    ●──────────●

		  顶点尖锐突出                    顶点外扩，折角变缓
		*/
		info := pointInfos[idx]
		currP := points[idx]
		if info.IsConvex && math.Abs(float64(info.Angle-float32(sharpAngleThresh))) < 1 {
			v1NormResult := norm(info.v1)

			v2NormResult := norm(info.v2)

			v1Norm := []float64{
				float64(info.v1.X) / v1NormResult,
				float64(info.v1.Y) / v1NormResult,
			}

			V2Norm := []float64{
				float64(info.v2.X) / v2NormResult,
				float64(info.v2.Y) / v2NormResult,
			}

			dirVec := []float64{
				v1Norm[0] + V2Norm[0],
				v1Norm[1] + V2Norm[1],
			}

			newNorm := normFloat64(dirVec[0], dirVec[1])

			d := (v1NormResult + v2NormResult) / 2
			newDirVec := []float64{
				(dirVec[0] / newNorm) * d,
				(dirVec[1] / newNorm) * d,
			}
			currP.X = int(math.Round(float64(currP.X) + newDirVec[0]))
			currP.Y = int(math.Round(float64(currP.Y) + newDirVec[1]))
		}

		res = append(res, currP)
	}

	return res
}

// 向量长度
func norm(p image.Point) float64 {
	return math.Sqrt(float64(p.X*p.X + p.Y*p.Y))
}

func normFloat64(x, y float64) float64 {
	return math.Sqrt(x*x + y*y)
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

	// 计算角度
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

	if err := gocv.BoxPoints(minRect, &ptsMat); err != nil {
		return nil
	}

	if ptsMat.Rows() != 4 || ptsMat.Cols() != 2 {
		return nil
	}

	// 读取4个点并计算中心
	quad := make([][2]float64, 4)
	var centerX, centerY float64
	for i := 0; i < 4; i++ {
		quad[i][0] = float64(ptsMat.GetFloatAt(i, 0))
		quad[i][1] = float64(ptsMat.GetFloatAt(i, 1))
		centerX += quad[i][0]
		centerY += quad[i][1]
	}
	centerX /= 4
	centerY /= 4

	// 按角度排序（逆时针）
	type pointInfo struct {
		idx   int
		angle float64
		x, y  float64
	}

	pts := make([]pointInfo, 4)
	for i := 0; i < 4; i++ {
		pts[i] = pointInfo{
			idx:   i,
			angle: math.Atan2(quad[i][1]-centerY, quad[i][0]-centerX),
			x:     quad[i][0],
			y:     quad[i][1],
		}
	}

	sort.Slice(pts, func(i, j int) bool {
		return pts[i].angle < pts[j].angle
	})

	// 找到左上角（x+y最小）
	topLeftIdx := 0
	minSum := pts[0].x + pts[0].y
	for i := 1; i < 4; i++ {
		if sum := pts[i].x + pts[i].y; sum < minSum {
			minSum = sum
			topLeftIdx = i
		}
	}

	// 构建结果（从左上角开始顺时针）
	result := make([]image.Point, 4)
	for i := 0; i < 4; i++ {
		src := pts[(topLeftIdx+i)%4]
		result[i] = image.Point{
			X: int(math.Round(src.x)),
			Y: int(math.Round(src.y)),
		}
	}

	return result
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
