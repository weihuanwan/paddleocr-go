package table

import ort "github.com/yalue/onnxruntime_go"

/*
*有线表格识别模型模型
 */
type WiredTableRecOnnxSession struct {
	OnnxSession     *ort.DynamicAdvancedSession
	Labels          []string // 标签字典 有线表格、无线表格
	targetShortEdge int
	CropSize        [2]int //  识方向模型裁剪大小 默认{224, 224}
	Alpha           [3]float32
	Beta            [3]float32
}
