package main

import (
	"fmt"
	"image"
	"image/color"
	"log"

	"github.com/weihuanwan/paddleocr-go/ocr"
	"github.com/weihuanwan/paddleocr-go/table"
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
		"test/model/wired_table_cell_det.onnx",
		[]string{"im_shape", "image", "scale_factor"},
		[]string{"fetch_name_0", "fetch_name_1"},
		options,
	)
	if err != nil {
		panic(err)
	}

	docLayoutSession := table.NewWiredTableCellsDetOnnxSession(layoutDetSessionInternal)

	imagePath := "test/images/table_recognition3.png"

	imageMat := gocv.IMRead(imagePath, gocv.IMReadColor)
	defer imageMat.Close()
	originImage1 := imageMat.Clone()
	layoutDetResults, err := docLayoutSession.Run(&imageMat)

	if err != nil {
		panic(err)
	}
	for i := 0; i < len(layoutDetResults); i++ {

		layoutDet := layoutDetResults[i]
		point := layoutDet.Point

		x1 := point[0]
		y1 := point[1]
		x2 := point[2]
		y2 := point[3]

		rect := image.Rect(x1, y1, x2, y2)

		// 画矩形
		gocv.Rectangle(&originImage1, rect, color.RGBA{255, 0, 0, 0}, 1)

		// 写标签
		label := fmt.Sprintf("%s %.2f", layoutDet.Label, layoutDet.Score)

		pt := image.Pt(x1, y1-5)
		gocv.PutText(&originImage1, label, pt,
			gocv.FontHersheySimplex,
			0.7,
			color.RGBA{0, 255, 0, 0},
			2)

		// 顺序号
		orderText := fmt.Sprintf("%d", layoutDet.Order)
		gocv.PutText(&originImage1, orderText,
			image.Pt(x1, y1-25),
			gocv.FontHersheySimplex,
			0.8,
			color.RGBA{255, 0, 255, 0},
			2)
	}
	gocv.IMWrite("layout_result.jpg", originImage1)

}
