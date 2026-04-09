package main

import (
	"fmt"
	"image"
	"image/color"
	"log"

	"github.com/weihuanwan/paddleocr-go/layout"
	"github.com/weihuanwan/paddleocr-go/ocr"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

func main() {

	// ✅ session options

	err := ocr.InitOrt("./test/lib/onnxruntime.dll")
	if err != nil {
		log.Fatalf("Error initializing Ort: %v", err)
	}

	options, _ := ort.NewSessionOptions()
	defer options.Destroy()
	// CLS
	layoutDetSessionInternal, err := ort.NewDynamicAdvancedSession(
		"C:\\Users\\Administrator\\.paddlex\\official_models\\PP-DocLayoutV3\\PP-DocLayoutV3.onnx",
		[]string{"im_shape", "image", "scale_factor"},
		[]string{"fetch_name_0", "fetch_name_1", "fetch_name_2"},
		options,
	)
	if err != nil {
		panic(err)
	}

	docLayoutSession := layout.NewLayoutDetSession(layoutDetSessionInternal)

	imagePath := "test/images/layout1.png"

	imageMat := gocv.IMRead(imagePath, gocv.IMReadColor)
	layoutDetResults, err := docLayoutSession.Run(&imageMat)
	if err != nil {
		panic(err)
	}

	//for i := 0; i < len(layoutDetResults); i++ {
	//
	//	layoutDet := layoutDetResults[i]
	//	point := layoutDet.Point
	//
	//	x1 := point[0]
	//	y1 := point[1]
	//	x2 := point[2]
	//	y2 := point[3]
	//
	//	rect := image.Rect(x1, y1, x2, y2)
	//
	//	// 画矩形
	//	gocv.Rectangle(&imageMat, rect, color.RGBA{255, 0, 0, 0}, 1)
	//
	//	// 写标签
	//	label := fmt.Sprintf("%s %.2f", layoutDet.Label, layoutDet.Score)
	//
	//	pt := image.Pt(x1, y1-5)
	//	gocv.PutText(&imageMat, label, pt,
	//		gocv.FontHersheySimplex,
	//		0.7,
	//		color.RGBA{0, 255, 0, 0},
	//		2)
	//
	//	// 顺序号
	//	orderText := fmt.Sprintf("%d", layoutDet.Order)
	//	gocv.PutText(&imageMat, orderText,
	//		image.Pt(x1, y1-25),
	//		gocv.FontHersheySimplex,
	//		0.8,
	//		color.RGBA{255, 0, 255, 0},
	//		2)
	//}

	//// 最后统一显示
	//w := gocv.NewWindow("layout")
	//w.IMShow(imageMat)
	//w.WaitKey(0)
	// 保存图片
	//gocv.IMWrite("layout_result.jpg", imageMat)
	for i := 0; i < len(layoutDetResults); i++ {

		layoutDet := layoutDetResults[i]
		//point := layoutDet.Point
		//
		//rect := image.Rect(point[0], point[1], point[2], point[3])
		//
		//cropImage := imageMat.Region(rect)

		cropImage := cropByBoxes(layoutDet, imageMat)
		name := fmt.Sprintf("%dlayout_result.jpg", i)

		gocv.IMWrite(name, cropImage)

	}

}

func cropByBoxes(layoutDet *layout.LayoutDetResult, imageMat gocv.Mat) gocv.Mat {
	point := layoutDet.Point

	rect := image.Rect(point[0], point[1], point[2], point[3])
	region := imageMat.Region(rect)
	cropImage := region.Clone() // 一定要 Clone！

	// 如果没有多边形，直接返回
	if layoutDet.PolygonPoints == nil || len(layoutDet.PolygonPoints) == 0 {
		return cropImage
	}

	// 1️⃣ 创建 mask（单通道）
	mask := gocv.NewMatWithSize(cropImage.Rows(), cropImage.Cols(), gocv.MatTypeCV8U)

	// 2️⃣ 构建 polygon（注意要减去 xmin, ymin）
	var pts []image.Point
	xmin := point[0]
	ymin := point[1]

	for _, p := range layoutDet.PolygonPoints {
		x := p.X - xmin
		y := p.Y - ymin
		pts = append(pts, image.Pt(x, y))
	}

	// 3️⃣ 填充多边形（mask=255 表示保留）
	pointsVector := gocv.NewPointsVectorFromPoints([][]image.Point{pts})
	gocv.FillPoly(&mask, pointsVector, color.RGBA{255, 255, 255, 0})

	// 4️⃣ 生成白色背景
	white := gocv.NewMatWithSizeFromScalar(
		gocv.NewScalar(255, 255, 255, 0),
		cropImage.Rows(),
		cropImage.Cols(),
		cropImage.Type(),
	)

	// 5️⃣ 用 mask 拷贝有效区域
	result := white.Clone()
	cropImage.CopyToWithMask(&result, mask)

	return result
}
