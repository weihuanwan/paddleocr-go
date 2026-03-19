package ocr

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"sort"

	clipper "github.com/cwbudde/go-clipper2/port"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

type DetOnnxSession struct {
	OnnxSession *ort.DynamicAdvancedSession
	Config      *PaddleOCRConfig
}

// Box 表示文本框（4个顶点）
type DetResult struct {
	Points []image.Point
	Score  float32
}

func (det DetOnnxSession) Run(originImage *gocv.Mat) ([]*DetResult, error) {
	// 缩放
	resizedImage, ratio, err := det.resize(originImage)

	if err != nil {
		return nil, err
	}

	defer resizedImage.Close()
	// 归一化
	imageNormalize, err := det.normalize(resizedImage)

	if err != nil {
		return nil, err
	}

	defer imageNormalize.Close()
	//w := gocv.NewWindow("image")
	//w.ResizeWindow(imageNormalize.Cols(), imageNormalize.Rows())
	//w.IMShow(*imageNormalize)
	//w.WaitKey(0)
	// 转换 chw
	imageCHW := HWCToCHW(imageNormalize)

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
	err = det.OnnxSession.Run([]ort.Value{detInputTensor}, []ort.Value{detOutputTensor})

	if err != nil {
		return nil, fmt.Errorf("run det OnnxSession error ", err)

	}
	return det.extractBoxes(detOutputTensor.GetData(), ratio)

}

/*
* 缩放
 */
func (det DetOnnxSession) resize(originImage *gocv.Mat) (*gocv.Mat, []int, error) {

	origHeight := originImage.Rows()
	origWidth := originImage.Cols()

	limitSideLen := det.Config.LimitSideLen
	maxSideLimit := det.Config.MaxSideLimit

	var ratio = 1.0
	if min(origHeight, origWidth) < limitSideLen {
		if origHeight < origWidth {
			ratio = float64(limitSideLen) / float64(origHeight)
		} else {
			ratio = float64(limitSideLen) / float64(origWidth)
		}
	}

	resizeH := int(float64(origHeight) * ratio)
	resizeW := int(float64(origWidth) * ratio)

	// 如果超过这限制
	if max(resizeH, resizeW) > maxSideLimit {
		ratio = float64(maxSideLimit) / float64(max(resizeH, resizeW))
		resizeH, resizeW = int(float64(resizeH)*ratio), int(float64(resizeW)*ratio)
	}

	resizeH = max(int(math.Round(float64(resizeH)/float64(32))*32), 32)

	resizeW = max(int(math.Round(float64(resizeW)/float64(32))*32), 32)

	// 如果等于
	if resizeH == origHeight && resizeW == origWidth {

		//克隆一份出来
		cloneImage := originImage.Clone()
		return &cloneImage, []int{origHeight, origWidth, origHeight, origWidth}, nil
	}

	if resizeH < 0 && resizeW < 0 {

		return nil, nil, fmt.Errorf("det resize resizeH 0 and resizeW 0")
	}

	resizedImage := gocv.NewMat()

	// 缩放
	err := gocv.Resize(*originImage,
		&resizedImage, image.Pt(resizeW, resizeH), 0, 0, gocv.InterpolationLinear)

	if err != nil {

		return nil, nil, fmt.Errorf("det resize error")

	}

	//ratioH := float64(resizeH) / float64(origHeight)
	//
	//ratioW := float64(resizeW) / float64(origWidth)

	return &resizedImage, []int{origHeight, origWidth, resizeH, resizeW}, nil

}

// 归一化处理
func (det DetOnnxSession) normalize(resizedImage *gocv.Mat) (*gocv.Mat, error) {
	c := resizedImage.Channels()

	// 获取rgb
	gbrSplit := gocv.Split(*resizedImage)

	scale := det.Config.Scale
	mean := det.Config.Mean
	std := det.Config.Std
	var alpha [3]float32
	var beta [3]float32

	for i := 0; i < c; i++ {
		alpha[i] = scale / std[i]
		beta[i] = -mean[i] / std[i]
	}

	for i := 0; i < c; i++ {
		cpMat := gocv.NewMat()
		old := gbrSplit[i]
		//转换 32 位
		err := old.ConvertTo(&cpMat, gocv.MatTypeCV32F)
		if err != nil {
			return nil, fmt.Errorf("det normalize  convert to 32f error")
		}
		old.Close()
		gbrSplit[i] = cpMat
		// 该通道乘以
		gbrSplit[i].MultiplyFloat(alpha[i])
		// 在加上
		gbrSplit[i].AddFloat(beta[i])

	}
	result := gocv.NewMat()

	err := gocv.Merge(gbrSplit, &result)
	if err != nil {

		return nil, fmt.Errorf("det normalize merge error")

	}
	return &result, nil
}

func unclip(box []image.Point, unclipRatio float32) gocv.PointVector {

	// 1️⃣ 先算面积和周长
	pv := gocv.NewPointVectorFromPoints(box)
	area := gocv.ContourArea(pv)
	length := gocv.ArcLength(pv, true)
	pv.Close()

	if length == 0 {
		return gocv.NewPointVector()
	}

	distance := area * float64(unclipRatio) / length

	// 2️⃣ 一定要 scale（核心）

	path64 := make(clipper.Path64, len(box))

	for i, p := range box {
		path64[i] = clipper.Point64{
			X: int64(p.X),
			Y: int64(p.Y),
		}
	}

	// 3️⃣ ArcTolerance 要跟 scale 对齐
	co := clipper.NewClipperOffset(2.0, 0.25)
	co.AddPath(path64, clipper.JoinRound, clipper.EndPolygon)

	solution, _ := co.Execute(distance)

	if len(solution) == 0 {
		return gocv.NewPointVector()
	}

	// 4️⃣ 转回 gocv.PointVector（只取第一个 polygon）
	result := gocv.NewPointVector()

	for _, pt := range solution[0] {
		result.Append(image.Pt(
			int(pt.X),
			int(pt.Y),
		))
	}

	return result
}

func getMiniBoxes(pv gocv.PointVector) ([]image.Point, float32) {
	// 获取最小面积 http://theailearner.com/2020/11/03/opencv-minimum-area-rectangle/ 好好看这个说明就知道四个位置了
	boundingBox := gocv.MinAreaRect(pv)

	boxPoints := gocv.NewMat()
	// 2️⃣ 获取 4 点 # 获取 四个顶点坐标 有时候顺时针，有时候逆时针，有时候从任意点开始
	err := gocv.BoxPoints(boundingBox, &boxPoints)
	if err != nil {
		panic("Error getting box points")
	}

	// 2️⃣ 转成 Go 结构
	sizeArr := boxPoints.Size()
	pts := make([]image.Point, sizeArr[0])
	// 遍历
	for i := 0; i < sizeArr[0]; i++ {
		x := boxPoints.GetFloatAt(i, 0)
		y := boxPoints.GetFloatAt(i, 1)
		// 取整数
		pts[i] = image.Pt(int(math.Ceil(float64(x))), int(math.Ceil(float64(y))))
	}
	// 4️⃣ 先按 X 排序（左右分组）
	sort.SliceStable(pts, func(i, j int) bool {
		return pts[i].X < pts[j].X
	})

	// 左边两个
	left := pts[:2]
	// 右边两个
	right := pts[2:]

	// 5️⃣ 左边按 Y 排（上/下）
	sort.Slice(left, func(i, j int) bool {
		return left[i].Y < left[j].Y
	})

	// 6️⃣ 右边按 Y 排
	sort.Slice(right, func(i, j int) bool {
		return right[i].Y < right[j].Y
	})
	// 7️⃣ 组成最终顺序
	// 左上 右上 右下 左下
	ordered := []image.Point{
		left[0],
		right[0],
		right[1],
		left[1],
	}
	width := boundingBox.Width
	height := boundingBox.Height

	shortSide := float32(math.Min(float64(width), float64(height)))
	return ordered, shortSide
}
func detResultFast(bitmap gocv.Mat, box []image.Point) float32 {

	h := bitmap.Rows()
	w := bitmap.Cols()

	// 1️⃣ 计算 xmin xmax ymin ymax（float）
	xmin, xmax := box[0].X, box[0].X

	ymin, ymax := box[0].Y, box[0].Y

	for i := 1; i < len(box); i++ {
		p := box[i]
		if p.X < xmin {
			xmin = p.X
		}
		if p.X > xmax {
			xmax = p.X
		}
		if p.Y < ymin {
			ymin = p.Y
		}
		if p.Y > ymax {
			ymax = p.Y
		}
	}

	// 2️⃣ floor / ceil + clamp（完全等价 python）
	xminInt := int(math.Max(0, math.Min(math.Floor(float64(xmin)), float64(w-1))))
	xmaxInt := int(math.Max(0, math.Min(math.Ceil(float64(xmax)), float64(w-1))))
	yminInt := int(math.Max(0, math.Min(math.Floor(float64(ymin)), float64(h-1))))
	ymaxInt := int(math.Max(0, math.Min(math.Ceil(float64(ymax)), float64(h-1))))

	// 3️⃣ (453,25)
	mask := gocv.Zeros(
		ymaxInt-yminInt+1,
		xmaxInt-xminInt+1,
		gocv.MatTypeCV8UC1,
	)
	defer mask.Close()

	// 4️⃣ 构造 polygon（减整数 xmin/ymin）
	pvs := gocv.NewPointsVector()
	defer pvs.Close()

	pv := gocv.NewPointVector()
	defer pv.Close()
	for _, p := range box {
		px := p.X - xminInt
		py := p.Y - yminInt

		pv.Append(image.Pt(px, py))
	}
	pvs.Append(pv)

	// opnecv 是bgr 格式不是rgb。这个写法对于python  cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
	err := gocv.FillPoly(&mask, pvs, color.RGBA{0, 0, 255, 0})
	if err != nil {
		panic(err)
	}
	//roiFloat := gocv.NewMat()
	//defer roiFloat.Close()
	//// 转换32 并且归一化
	//errbitmap := bitmap.ConvertToWithParams(&roiFloat, gocv.MatTypeCV32F, 1.0/255.0, 0)
	//
	//if errbitmap != nil {
	//	panic(errbitmap)
	//}
	// 剪切
	rect := image.Rect(
		xminInt,
		yminInt,
		xmaxInt+1,
		ymaxInt+1,
	)
	roi := bitmap.Region(rect)

	defer roi.Close()
	// 7️⃣ mean
	scalar := roi.MeanWithMask(mask)
	// 由于 bitmap 没有转换32f 并且归一化 返回的是 0-255 之间，不是0-1。所以这边获取数据时候需要 / 255.0 得到 0-1，这样会快一些
	return float32(scalar.Val1 / 255.0)
}

func getMinAreaRectCrop(originImage *gocv.Mat, box *DetResult) *gocv.Mat {
	points := box.Points

	pv := gocv.NewPointVectorFromPoints(points)
	defer pv.Close()
	boundingBox := gocv.MinAreaRect(pv)

	boxPoints := gocv.NewMat()
	// 2️⃣ 获取矩形四个顶点
	err := gocv.BoxPoints(boundingBox, &boxPoints)
	if err != nil {
		panic("Error getting box points")
	}

	// 2️⃣ 转成 Go 结构
	sizeArr := boxPoints.Size()
	pts := make([]image.Point, sizeArr[0])
	// 遍历
	for i := 0; i < sizeArr[0]; i++ {
		x := boxPoints.GetFloatAt(i, 0)
		y := boxPoints.GetFloatAt(i, 1)
		pts[i] = image.Pt(int(math.Ceil(float64(x))), int(math.Ceil(float64(y))))
	}
	// 4️⃣ 先按 X 排序（左右分组）
	sort.SliceStable(pts, func(i, j int) bool {
		return pts[i].X < pts[j].X
	})

	indexA := 0
	indexB := 1
	indexC := 2
	indexD := 3
	if pts[1].Y > pts[0].Y {
		indexA = 0
		indexD = 1
	} else {
		indexA = 1
		indexD = 0
	}

	if pts[3].Y > pts[2].Y {
		indexB = 2
		indexC = 3
	} else {
		indexB = 3
		indexC = 2
	}
	sortPoints := []image.Point{
		pts[indexA], pts[indexB], pts[indexC], pts[indexD],
	}

	return getRotateCropImage(originImage, sortPoints)
}

func getRotateCropImage(originImage *gocv.Mat, points []image.Point) *gocv.Mat {
	imgCropWidth := int(math.Ceil(math.Max(
		distance(points[0], points[1]),
		distance(points[2], points[3]),
	)))

	imgCropHeight := int(math.Ceil(math.Max(
		distance(points[0], points[3]),
		distance(points[1], points[2]),
	)))

	dstPV := gocv.NewPointVector()
	defer dstPV.Close()
	dstPV.Append(image.Pt(0, 0))
	dstPV.Append(image.Pt(imgCropWidth, 0))
	dstPV.Append(image.Pt(imgCropWidth, imgCropHeight))
	dstPV.Append(image.Pt(0, imgCropHeight))

	srcPV := gocv.NewPointVectorFromPoints(points)
	defer srcPV.Close()
	dstImg := gocv.NewMat()

	m := gocv.GetPerspectiveTransform(srcPV, dstPV)

	err := gocv.WarpPerspectiveWithParams(*originImage, &dstImg, m, image.Point{imgCropWidth, imgCropHeight},
		gocv.InterpolationCubic, gocv.BorderReplicate, color.RGBA{0, 0, 0, 255})
	if err != nil {
		panic("Error getting dst image")
	}
	dstImgHeight, dstImgWidth := dstImg.Rows(), dstImg.Cols()
	//w := gocv.NewWindow("image")
	//w.ResizeWindow(dstImgWidth, dstImgHeight)
	//w.IMShow(dstImg)
	//w.WaitKey(0)
	/**
	判断宽度
		ppocr 默认是 1.5
		目前遇到如果只有一个字就会出现不旋转的情况，所以这边给1.25, 根据自己需求来设置
	*/
	if (float32(dstImgHeight)*1.0)/float32(dstImgWidth) >= 1.25 {
		rotated := gocv.NewMat()
		// 旋转90度
		err := gocv.Rotate(dstImg, &rotated, gocv.Rotate90CounterClockwise)
		if err != nil {
			defer rotated.Close()
			panic("Error rotating dst image")
		}
		dstImg.Close()
		return &rotated
	}
	return &dstImg
}
func distance(p1, p2 image.Point) float64 {
	dx := p1.X - p2.X
	dy := p1.Y - p2.Y
	return math.Sqrt(float64(dx*dx + dy*dy))
}
func (det DetOnnxSession) extractBoxes(heatmap []float32, ratio []int) ([]*DetResult, error) {
	originHeight := ratio[0]
	originWidth := ratio[1]
	resizeHeight := ratio[2]
	resizeWidth := ratio[3]

	heightScale := float32(originHeight) / float32(resizeHeight)

	widthScale := float32(originWidth) / float32(resizeWidth)

	if len(heatmap) != resizeHeight*resizeWidth {
		return nil, fmt.Errorf("det heatmap size mismatch")
	}
	heatmapMat := gocv.NewMatWithSize(resizeHeight, resizeWidth, gocv.MatTypeCV8UC1)
	defer heatmapMat.Close()
	for y := 0; y < resizeHeight; y++ {
		for x := 0; x < resizeWidth; x++ {
			index := y*resizeWidth + x
			// 超过这个阈值设置为白色
			if heatmap[index] > det.Config.HeatmapThreshold {

				heatmapMat.SetUCharAt(y, x, 255)
			}
		}
	}

	//w := gocv.NewWindow("image")
	//w.ResizeWindow(resizeWidth, resizeHeight)
	//w.IMShow(heatmapMat)
	//w.WaitKey(0)
	// 查找轮廓
	pvs := gocv.FindContours(heatmapMat, gocv.RetrievalList, gocv.ChainApproxSimple)

	boxlist := make([]*DetResult, 0)

	for i := 0; i < pvs.Size(); i++ {
		pv := pvs.At(i)

		p, side := getMiniBoxes(pv)

		if side < det.Config.MinSize {
			continue
		}
		score := detResultFast(heatmapMat, p)

		if score < det.Config.BoxThresh {
			continue
		}
		// 扩充
		pv2 := unclip(p, det.Config.UnclipRatio)

		boxs, side := getMiniBoxes(pv2)
		if side < det.Config.MinSize+2 {
			continue
		}

		result := make([]image.Point, 0, 4)
		for i := 0; i < len(boxs); i++ {
			box := boxs[i]
			box.X = int(min(float32(box.X)*widthScale, float32(originWidth)))
			box.Y = int(min(float32(box.Y)*heightScale, float32(originHeight)))
			result = append(result, box)
		}
		boxlist = append(boxlist, &DetResult{result, score})
	}
	return boxlist, nil
}
