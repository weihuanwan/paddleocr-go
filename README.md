paddleocr-go 是一个面向 Go 生态的高性能 OCR 工具库，基于 ONNX Runtime 与 GOCV 构建，深度集成 PaddleOCR（PPOCR）模型能力。
提供从方向分类、文本检测、文本识别的完整流程封装，开箱即用，兼顾性能与扩展性，适用于服务端 OCR、自动化处理及高并发场景

# 环境准备

**注意**：按照 gocv 官方文档安装的 MinGW-W64 与 CMake 环境会与 `github.com/gen2brain/go-fitz` 依赖产生冲突。因此，请勿参照 gocv 官网指南进行安装，必须统一使用 MSYS2 来安装 MinGW-W64 和 CMake。

1. 安装 msys2 和 编译 gocv
   - msys2 地址：https://www.msys2.org
   
     1. 打开 ucrt64.exe 命令窗口
   
        - pacman -Syu
        - pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-make
   
     2. 验证
   
        > gcc --version
        > cmake --version
   
   - gocv 地址 ：https://gocv.io/getting-started
   
     - .\win_download_opencv.cmd
   
     - .\win_build_opencv.cmd
   
     - go run cmd\version\main.go
   
       > gocv version: 0.43.0
       > opencv lib version: 4.13.0
   
3. 下载 onnxruntime
   - 地址：https://github.com/microsoft/onnxruntime
   - 版本：onnxruntime-xxx-1.29.0
   
4. 准备 PaddleOCR ONNX  模型
   - 地址1：https://huggingface.co/weihuanwan
# 快速开始

1. 下载依赖 

   - go get -u github.com/weihuanwan/paddleocr-go

2. 代码示例

   ```go
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
   		"./lib/onnxruntime.dll",
   		"./model/det.onnx",
   		"./model/rec.onnx",
   		"./model/cls.onnx",
   		"./model/dict.txt",
   		false,
   		true,
   	)
   
   	session, err := ocr.NewPaddleOCRSession(config)
   	if err != nil {
   		log.Fatalf("create ocr session: %v\n", err)
   	}
   
   	defer session.Destroy()
   
   	imagePath := "https://github.com/weihuanwan/paddle-ocr-model/blob/main/test.jpg?raw=true"
   
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
   
   ```
   

# 示例效果

|                             原图                             |                            检测图                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](https://github.com/weihuanwan/paddle-ocr-model/blob/main/test.jpg?raw=true) | ![](https://github.com/weihuanwan/paddle-ocr-model/blob/main/det_result_test.jpg?raw=true) |
