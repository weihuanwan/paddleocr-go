package vl

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"runtime"
	"sync"

	"github.com/weihuanwan/paddleocr-go/common"
	"github.com/weihuanwan/paddleocr-go/layout"
	"github.com/weihuanwan/paddleocr-go/utils"
	"gocv.io/x/gocv"
)

type PaddleOCRVL struct {
	Model  string            // 模型名称
	Url    string            // 请求路径
	ApiKey string            // 请求路径
	Tasks  map[string]string //任务类型

	LayoutDetSession *layout.LayoutDetSession //版面分析模型
}

func NewPaddleOCRVLChatCompletionRequest(modelName string,
	dataURL string,
	task string) ChatCompletionRequest {
	messages := []Messages{
		{
			Role: "user",
			Content: []Content{
				{
					Type: "image_url",
					ImageURL: &ImageURL{
						URL: dataURL,
					},
				},
				{
					Type: "text",
					Text: task,
				},
			},
		},
	}
	return ChatCompletionRequest{
		Model:       modelName,
		Messages:    messages,
		Temperature: 0.0,
	}
}

func NewDefaultPaddleOCRVL(
	model string,
	url string,
	apiKey string,
	layoutDetSession *layout.LayoutDetSession,

) *PaddleOCRVL {
	tasks := map[string]string{
		"ocr":      "OCR:",
		"table":    "Table Recognition:",
		"formula":  "Formula Recognition:",
		"chart":    "Chart Recognition:",
		"seal":     "Seal Recognition:",
		"spotting": "Spotting:",
	}
	paddleOCRVL := &PaddleOCRVL{
		model,
		url,
		apiKey,
		tasks,
		layoutDetSession,
	}

	return paddleOCRVL
}

type PaddleOCRVLBlock struct {
	*common.LayoutDetResult
	Text string
}

// PageResult 封装单页 OCR 结果
type PageResult struct {
	PageIndex int                 // 页码（从 0 开始）
	Mat       gocv.Mat            // 该页图像（调用方负责最终 Close）
	Blocks    []*PaddleOCRVLBlock // OCR 识别结果
	Err       error               // 该页处理过程中的错误
}

// Close 释放该页 Mat
func (p *PageResult) Close() {
	p.Mat.Close()
}

func (session *PaddleOCRVL) Close(pages []*PageResult) {
	for _, p := range pages {
		if p != nil {
			p.Close()
		}
	}
}

// RunOCR 加载并并发 OCR，返回按页码排序的结果
func (session *PaddleOCRVL) RunOCR(imagePath string) ([]*PageResult, error) {
	mats, name, err := utils.Load(imagePath)
	if err != nil {
		return nil, fmt.Errorf("load %s failed: %w", name, err)
	}

	n := len(mats)
	if n == 0 {
		return nil, fmt.Errorf("no pages loaded from %s", imagePath)
	}

	results := make([]*PageResult, n)

	// 信号量控制并发数，防止多页 PDF 同时渲染爆内存
	workers := runtime.NumCPU()
	if n < workers {
		workers = n
	}
	sem := make(chan struct{}, workers)

	var wg sync.WaitGroup
	for i := range mats {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			sem <- struct{}{}        // 获取令牌
			defer func() { <-sem }() // 释放令牌

			blocks, err := session.runOCR(&mats[idx])
			results[idx] = &PageResult{
				PageIndex: idx,
				Mat:       mats[idx],
				Blocks:    blocks,
				Err:       err,
			}
		}(i)
	}
	wg.Wait()

	// 检查是否有页出错，出错时统一清理所有已分配的 Mat
	var firstErr error
	for _, r := range results {
		if r != nil && r.Err != nil && firstErr == nil {
			firstErr = fmt.Errorf("page %d OCR failed: %w", r.PageIndex, r.Err)
		}
	}
	if firstErr != nil {
		for _, r := range results {
			if r != nil {
				r.Mat.Close()
			}
		}
		return nil, firstErr
	}

	return results, nil
}

func (session *PaddleOCRVL) runOCR(originImage *gocv.Mat) ([]*PaddleOCRVLBlock, error) {
	// 版面分析模型识别
	layoutDetResult, err := session.LayoutDetSession.Run(originImage)
	if err != nil {
		return nil, err
	}
	paddleOCRVLBlocks := session.getLayoutParsingResults(layoutDetResult, originImage)
	return paddleOCRVLBlocks, nil
}
func (session *PaddleOCRVL) RunOCRFromMat(originImage *gocv.Mat) ([]*PaddleOCRVLBlock, error) {
	// 版面分析模型识别
	layoutDetResult, err := session.LayoutDetSession.Run(originImage)
	if err != nil {
		return nil, err
	}
	paddleOCRVLBlocks := session.getLayoutParsingResults(layoutDetResult, originImage)
	return paddleOCRVLBlocks, nil
}

func (session *PaddleOCRVL) getLayoutParsingResults(
	layoutDetResult []*common.LayoutDetResult,
	originImage *gocv.Mat,
) []*PaddleOCRVLBlock {

	filterLayoutDetResult := filterOverlapBoxes(layoutDetResult, "auto")
	final := make([]*PaddleOCRVLBlock, 0, len(filterLayoutDetResult))
	for i := 0; i < len(filterLayoutDetResult); i++ {
		detResult := filterLayoutDetResult[i]

		cropImage, err := common.CropByBoxes(detResult, originImage)

		base64Str, err := MatToBase64(cropImage)
		cropImage.Close()
		req := NewPaddleOCRVLChatCompletionRequest(
			session.Model,
			base64Str,
			session.getTask(detResult.Label),
		)

		resp, err := session.Run(req)
		if err != nil {
			log.Printf("Error PaddleOCRVL Run: %v", err)
			continue
		}

		ocrResult := resp.Choices[0].Message.Content
		text := ocrResult

		if detResult.Label == "table" {
			text = utils.ConvertOtslToHtml(ocrResult)
		}

		block := &PaddleOCRVLBlock{
			LayoutDetResult: detResult,
			Text:            text,
		}
		final = append(final, block)

	}

	return final
}

func (session *PaddleOCRVL) getTask(key string) string {
	task := session.Tasks[key]
	if task == "" {
		task = "OCR:"
	}
	return task
}
func MatToBase64(mat *gocv.Mat) (string, error) {

	buf, err := gocv.IMEncode(".png", *mat)
	if err != nil {
		return "", err
	}
	defer buf.Close()
	// 2. 转 base64
	base64Str := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.GetBytes())
	return base64Str, nil
}
func (session *PaddleOCRVL) Run(request ChatCompletionRequest) (*ChatCompletionResponse, error) {

	request.Model = session.Model
	reqBody, err := json.Marshal(request)
	// 5️⃣ 构造 HTTP 请求（标准写法）
	req, err := http.NewRequest(
		"POST",
		session.Url,
		bytes.NewBuffer(reqBody),
	)
	if err != nil {
		return nil, fmt.Errorf("req error %s  %s", session.Url, err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", session.ApiKey)

	// 6️⃣ 发请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send api error %s  %s", session.Url, err)
	}
	defer resp.Body.Close()

	// 7️⃣ 读取返回
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body error %s  %s", session.Url, err)
	}

	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("请求失败")
	}
	var result ChatCompletionResponse
	err = json.Unmarshal(body, &result)

	return &result, nil

}
