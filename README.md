paddle-ocr-go 是一款基于 Golang + ONNX + GOCV 构建的 OCR 工具库，专注于为 Go 生态提供简单易用、可扩展的文字识别能力。 目前已完成与 PaddleOCR 的对接，支持快速实现图像文字检测与识别。

# 环境准备

1. 安装 gocv
   - 地址 ：https://gocv.io/getting-started
2. 下载 onnx
   - 地址：https://github.com/microsoft/onnxruntime
   - 版本：onnxruntime-xxx-1.24.1
3. 准备 PaddleOCR onnx 模型

# 快速开始

1. 下载依赖 
   - go get -u github.com/weihuanwan/paddle-ocr-go

