package main

import (
	"fmt"

	"image"
	"log"
	"sort"
	"time"

	"github.com/weihuanwan/paddleocr-go/ocr"
)

type OCRItem struct {
	Points []image.Point
	Text   string
}

func main() {
	// 开始计时
	start := time.Now()
	config := ocr.NewDefaultPaddleOCRConfig(
		"test/lib/onnxruntime.dll",
		"test/model/det.onnx",
		"test/model/rec.onnx",
		"test/model/cls.onnx",
		"test/model/dict.txt",
		false,
		true,
	)

	session, err := ocr.NewPaddleOCRSession(config)
	if err != nil {
		log.Fatalf("create ocr session: %v\n", err)
	}

	defer session.Destroy()

	imagePath := "test/images/test.jpg"

	// 检测
	ocrResult, _ := session.RunOCR(imagePath)

	// 结束计时
	cost := time.Since(start)
	log.Printf("OCR耗时: %v\n", cost)

	// 创建一个结构体来存储配对的结果
	items := make([]OCRItem, len(ocrResult.DetResult))
	for i := 0; i < len(ocrResult.DetResult); i++ {
		items[i] = OCRItem{
			Points: ocrResult.DetResult[i].Points,
			Text:   ocrResult.RecResult[i].Text,
		}
	}

	// 按Y坐标从上到下排序（取每个检测框的左上角Y坐标）
	sort.Slice(items, func(i, j int) bool {
		return items[i].Points[0].Y < items[j].Points[0].Y // 按Y坐标升序
	})
	items = sortOCR(items)
	// 打印排序后的结果
	for i, item := range items {
		fmt.Printf("%d. Y=%d: %s\n", i+1, centerY(item.Points), item.Text)
		fmt.Printf("   坐标: %v\n", item.Points)
	}

}
func centerY(points []image.Point) int {
	return (points[0].Y + points[2].Y) / 2
}

func centerX(points []image.Point) int {
	return (points[0].X + points[2].X) / 2
}

func abs(a int) int {
	if a < 0 {
		return -a
	}
	return a
}

func sortOCR(items []OCRItem) []OCRItem {

	// 1️⃣ 先按中心Y排序
	sort.Slice(items, func(i, j int) bool {
		return centerY(items[i].Points) < centerY(items[j].Points)
	})

	var lines [][]OCRItem
	var current []OCRItem

	// 行阈值（根据你的图片大小可以调）
	threshold := 25

	for _, item := range items {

		if len(current) == 0 {
			current = append(current, item)
			continue
		}

		if abs(centerY(item.Points)-centerY(current[0].Points)) < threshold {
			current = append(current, item)
		} else {
			lines = append(lines, current)
			current = []OCRItem{item}
		}
	}

	if len(current) > 0 {
		lines = append(lines, current)
	}

	// 2️⃣ 每一行按X排序
	var result []OCRItem

	for _, line := range lines {

		sort.Slice(line, func(i, j int) bool {
			return centerX(line[i].Points) < centerX(line[j].Points)
		})

		result = append(result, line...)
	}

	return result
}
