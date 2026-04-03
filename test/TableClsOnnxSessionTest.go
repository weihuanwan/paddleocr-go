package main

import (
	"fmt"
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
		"C:\\Users\\Administrator\\.paddlex\\official_models\\PP-LCNet_x1_0_table_cls\\inference.onnx",
		[]string{"x"},
		[]string{"fetch_name_0"},
		options,
	)
	if err != nil {
		panic(err)
	}

	docLayoutSession := table.NewTableClsOnnxSession(layoutDetSessionInternal)

	imagePath := "D:\\workspaces\\paddleocr-go\\ 11 layout_result.jpg"

	imageMat := gocv.IMRead(imagePath, gocv.IMReadColor)
	layoutDetResults, err := docLayoutSession.Run(&imageMat)
	if err != nil {
		panic(err)
	}

	fmt.Println(layoutDetResults)

}
