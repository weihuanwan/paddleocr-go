package main

import (
	"fmt"
	"log"

	"github.com/weihuanwan/paddleocr-go/layout"
	"github.com/weihuanwan/paddleocr-go/ocr"
	ort "github.com/yalue/onnxruntime_go"
)

func main() {

	err := ocr.InitOrt("./test/lib/onnxruntime.dll")
	if err != nil {
		log.Fatalf("Error initializing Ort: %v", err)
	}

	options, _ := ort.NewSessionOptions()
	defer options.Destroy()
	// CLS
	layoutDetSessionInternal, err := ort.NewDynamicAdvancedSession(
		"test/model/PP-DocLayoutV3.onnx",
		[]string{"im_shape", "image", "scale_factor"},
		[]string{"fetch_name_0", "fetch_name_1", "fetch_name_2"},
		options,
	)
	if err != nil {
		panic(err)
	}

	docLayoutSession := layout.NewLayoutDetSession(layoutDetSessionInternal)

	paddleOCRVL := ocr.NewDefaultPaddleOCRVL("PaddlePaddle/PaddleOCR-VL-1.5",
		"http://localhost:8000/v1/chat/completions", docLayoutSession)

	imagePath := "test/images/word.png"
	//
	//imageMat := gocv.IMRead(imagePath, gocv.IMReadColor)
	//defer imageMat.Close()

	paddleOCRVLBlocks, err := paddleOCRVL.RunOCR(imagePath)

	if err != nil {
		panic(err)
	}
	for i := 0; i < len(paddleOCRVLBlocks); i++ {
		block := paddleOCRVLBlocks[i]
		fmt.Println(block.Label)
		fmt.Println(block.Text)
		fmt.Println("----------------------------")
	}

	// 8️⃣ 输出结果
	//fmt.Println(result)
}
