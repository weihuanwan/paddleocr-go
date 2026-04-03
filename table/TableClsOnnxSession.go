package table

import ort "github.com/yalue/onnxruntime_go"

/*
*表格分类模型
 */
type TableClsOnnxSession struct {
	OnnxSession *ort.DynamicAdvancedSession
}
