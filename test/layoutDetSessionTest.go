package main

import (
	"log"

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

	docLayoutSession := ocr.NewLayoutDetSession(layoutDetSessionInternal)

	imagePath := "D:\\workspaces\\paddleocr-go\\huqi.png"

	imageMat := gocv.IMRead(imagePath, gocv.IMReadColor)
	docLayoutSession.Run(&imageMat)

}
