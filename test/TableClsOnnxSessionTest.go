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
		"test/model/table_cls.onnx",
		[]string{"x"},
		[]string{"fetch_name_0"},
		options,
	)
	if err != nil {
		panic(err)
	}

	tableClsSession := table.NewTableClsOnnxSession(layoutDetSessionInternal)

	//imagePath := "test/images/table_recognition1.jpg"
	imagePath := "test/images/table_recognition2.jpg"

	imageMat := gocv.IMRead(imagePath, gocv.IMReadColor)
	tableClsResult, err := tableClsSession.Run(&imageMat)
	if err != nil {
		panic(err)
	}

	fmt.Println(tableClsResult)

}
