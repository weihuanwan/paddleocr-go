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

		// 写标签 - 字体更小，显示在区域中间
		label := fmt.Sprintf("%s %.2f", layoutDet.Label, layoutDet.Score)

		// 计算文本尺寸以居中显示
		textSize := gocv.GetTextSize(label, gocv.FontHersheySimplex, 0.5, 1)

		// 计算矩形中心
		centerX := (x1 + x2) / 2
		centerY := (y1 + y2) / 2

		// 文本左上角坐标（基于中心点偏移）
		textX := centerX - textSize.X/2
		textY := centerY + textSize.Y/2

		pt := image.Pt(textX, textY)

		// 字体大小 0.5，线条粗细 1
		gocv.PutText(&originImage1, label, pt,
			gocv.FontHersheySimplex,
			0.5, // 字体缩放因子，原来是 0.7
			color.RGBA{0, 255, 0, 0},
			1)
	}
	gocv.IMWrite("layout_result.jpg", originImage1)

}
