package main

import (
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
	layoutDetResults, err := docLayoutSession.Run(&imageMat)

	if err != nil {
		panic(err)
	}
	for i := 0; i < len(layoutDetResults); i++ {

	}

}
