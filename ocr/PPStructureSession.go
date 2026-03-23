package ocr

import (
	"fmt"

	"github.com/weihuanwan/paddleocr-go/common"
	ort "github.com/yalue/onnxruntime_go"
)

type PPStructureSession struct {
	DocLayoutSession *LayoutDetSession // 方向
}

/*
*
ppocr 配置
*/
type PPStructureOCRConfig struct {
	*PaddleOCRConfig
	DocLayoutPlusLModelPath string
}

func NewDefaultPPStructureOCRConfig(
	onnxRuntimeLibPath string,
	docLayoutPlusLModelPath string,

) *PPStructureOCRConfig {

	paddleOCRConfig := NewDefaultPaddleOCRConfig(
		onnxRuntimeLibPath,
		"./test/model/det.onnx",
		"./test/model/rec.onnx",
		"./test/model/cls.onnx",
		"./test/model/dict.txt",
		false,
		true,
	)
	config := &PPStructureOCRConfig{
		PaddleOCRConfig:         paddleOCRConfig,
		DocLayoutPlusLModelPath: docLayoutPlusLModelPath,
	}

	return config
}
func NewPPStructureSession(config *PPStructureOCRConfig) (*PPStructureSession, error) {

	// ✅ 基础校验
	if config.OnnxRuntimeLibPath == "" {
		return nil, fmt.Errorf("OnnxRuntimeLibPath is required")
	}

	// ✅ 初始化 ONNX（只执行一次）
	if err := initOrt(config.OnnxRuntimeLibPath); err != nil {
		return nil, fmt.Errorf("initialize ONNX Runtime: %w", err)
	}

	// ✅ session options
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("create session options: %w", err)
	}
	defer options.Destroy()

	if config.NumThreads > 0 {
		if err := options.SetIntraOpNumThreads(config.NumThreads); err != nil {
			return nil, fmt.Errorf("set intra-op threads: %w", err)
		}
	}
	// ✅ CUDA
	if config.UseCuda {
		cudaOptions, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return nil, fmt.Errorf("create CUDA provider options: %w", err)
		}
		defer cudaOptions.Destroy()

		if err := options.AppendExecutionProviderCUDA(cudaOptions); err != nil {
			return nil, fmt.Errorf("append CUDA provider: %w", err)
		}
	}

	// CLS
	docLayoutPlusLSessionInternal, err := ort.NewDynamicAdvancedSession(
		config.DocLayoutPlusLModelPath,
		[]string{"im_shape", "image", "scale_factor"},
		[]string{"fetch_name_0", "fetch_name_1", "fetch_name_2"},
		options,
	)
	if err != nil {
		return nil, fmt.Errorf("create classification session: %w", err)
	}
	docLayoutSession := NewLayoutDetSession(docLayoutPlusLSessionInternal)
	// 构建 session
	session := &PPStructureSession{
		DocLayoutSession: docLayoutSession,
	}

	return session, nil
}
func (session *PPStructureSession) RunOCR(imagePath string) error {
	imageMat, _, err := common.LoadImage(imagePath)

	if err != nil {

		return err
	}

	session.DocLayoutSession.Run(imageMat)

	return nil
}
