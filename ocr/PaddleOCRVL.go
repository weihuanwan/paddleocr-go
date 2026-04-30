package ocr

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sync"

	"github.com/weihuanwan/paddleocr-go/common"
	"github.com/weihuanwan/paddleocr-go/layout"
	"gocv.io/x/gocv"
)

type PaddleOCRVL struct {
	Model string            // 模型名称
	Url   string            // 请求路径
	Tasks map[string]string //任务类型

	LayoutDetSession *layout.LayoutDetSession //版面分析模型
}

type ChatCompletionRequest struct {
	Model       string     `json:"model"`
	Messages    []Messages `json:"messages"`
	Temperature float32    `json:"temperature"`
}

func NewChatCompletionRequest(modelName string,
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

type Messages struct {
	Role    string    `json:"role"`
	Content []Content `json:"content"`
}

type Content struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *ImageURL `json:"image_url,omitempty"`
}

type ImageURL struct {
	URL string `json:"url"`
}

type ChatCompletionResponse struct {
	ID                string      `json:"id"`
	Object            string      `json:"object"`
	Created           int64       `json:"created"`
	Model             string      `json:"model"`
	Choices           []Choice    `json:"choices"`
	ServiceTier       string      `json:"service_tier"`
	SystemFingerprint string      `json:"system_fingerprint"`
	Usage             Usage       `json:"usage"`
	PromptLogprobs    interface{} `json:"prompt_logprobs"`
	PromptTokenIDs    interface{} `json:"prompt_token_ids"`
	KVTransferParams  interface{} `json:"kv_transfer_params"`
}

type Choice struct {
	Index        int         `json:"index"`
	Message      Message     `json:"message"`
	Logprobs     interface{} `json:"logprobs"`
	FinishReason string      `json:"finish_reason"`
	StopReason   string      `json:"stop_reason"`
	TokenIDs     interface{} `json:"token_ids"`
}

type Message struct {
	Role         string        `json:"role"`
	Content      string        `json:"content"`
	Refusal      string        `json:"refusal"`
	Annotations  interface{}   `json:"annotations"`
	Audio        interface{}   `json:"audio"`
	FunctionCall interface{}   `json:"function_call"`
	ToolCalls    []interface{} `json:"tool_calls"`
	Reasoning    interface{}   `json:"reasoning"`
}

type Usage struct {
	PromptTokens        int         `json:"prompt_tokens"`
	TotalTokens         int         `json:"total_tokens"`
	CompletionTokens    int         `json:"completion_tokens"`
	PromptTokensDetails interface{} `json:"prompt_tokens_details"`
}

func NewDefaultPaddleOCRVL(
	model string,
	url string,
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
		tasks,
		layoutDetSession,
	}

	return paddleOCRVL
}

type PaddleOCRVLBlock struct {
	*common.LayoutDetResult

	OcrResult string
	Text      string
}

func (session *PaddleOCRVL) RunOCR(imagePath string) ([]*PaddleOCRVLBlock, error) {
	originImage, _, err := common.LoadImage(imagePath)
	if err != nil {
		return nil, err
	}
	defer originImage.Close()
	// 版面分析模型识别
	layoutDetResult, err := session.LayoutDetSession.Run(originImage)
	if err != nil {
		return nil, err
	}
	paddleOCRVLBlocks := session.getLayoutParsingResults(layoutDetResult, originImage)
	return paddleOCRVLBlocks, nil
}

type layoutTask struct {
	index     int
	detResult *common.LayoutDetResult
}

type layoutResult struct {
	index int
	block *PaddleOCRVLBlock
	err   error
}

func (session *PaddleOCRVL) getLayoutParsingResults(
	layoutDetResult []*common.LayoutDetResult,
	originImage *gocv.Mat,
) []*PaddleOCRVLBlock {

	filterLayoutDetResult := filterOverlapBoxes(layoutDetResult, "auto")

	taskCh := make(chan layoutTask, len(filterLayoutDetResult))
	resultCh := make(chan layoutResult, len(filterLayoutDetResult))

	workerNum := 8 // 👈 可调：CPU + 网络决定（一般 4~16）

	var wg sync.WaitGroup

	// worker
	for w := 0; w < workerNum; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for task := range taskCh {

				detResult := task.detResult

				cropImage, err := common.CropByBoxes(detResult, originImage)
				if err != nil {
					resultCh <- layoutResult{index: task.index, err: err}
					continue
				}

				base64Str, err := MatToBase64(cropImage)
				if err != nil {
					resultCh <- layoutResult{index: task.index, err: err}
					continue
				}

				req := NewChatCompletionRequest(
					session.Model,
					base64Str,
					session.getTask(detResult.Label),
				)

				resp, err := session.Run(req)
				if err != nil {
					resultCh <- layoutResult{index: task.index, err: err}
					continue
				}

				ocrResult := resp.Choices[0].Message.Content
				text := ocrResult

				if detResult.Label == "table" {
					text = ConvertOtslToHtml(ocrResult)
				}

				block := &PaddleOCRVLBlock{
					LayoutDetResult: detResult,
					OcrResult:       ocrResult,
					Text:            text,
				}

				resultCh <- layoutResult{
					index: task.index,
					block: block,
				}
			}
		}()
	}

	// 投递任务
	go func() {
		for i, det := range filterLayoutDetResult {
			taskCh <- layoutTask{
				index:     i,
				detResult: det,
			}
		}
		close(taskCh)
	}()

	// 等待 worker 结束
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// 收集结果（保序）
	results := make([]*PaddleOCRVLBlock, len(filterLayoutDetResult))

	for r := range resultCh {
		if r.err != nil {
			fmt.Printf("error index=%d err=%v\n", r.index, r.err)
			continue
		}
		results[r.index] = r.block
	}

	// 去 nil
	final := make([]*PaddleOCRVLBlock, 0, len(results))
	for _, v := range results {
		if v != nil {
			final = append(final, v)
		}
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
func filterOverlapBoxes(results []*common.LayoutDetResult, layoutShapeMode string) []*common.LayoutDetResult {

	// 1️⃣ 过滤掉 reference（和 Python 一样）
	boxes := make([]*common.LayoutDetResult, 0)
	for _, r := range results {
		if r.Label != "reference" {
			boxes = append(boxes, r)
		}
	}

	// 2️⃣ set
	dropped := make(map[int]struct{})

	for i := 0; i < len(boxes); i++ {

		pointI := boxes[i].Point
		x1, y1, x2, y2 := pointI[0], pointI[1], pointI[2], pointI[3]
		w, h := x2-x1, y2-y1

		// Python: 直接 add，但不 continue
		if w < 6 || h < 6 {
			dropped[i] = struct{}{}
		}

		for j := i + 1; j < len(boxes); j++ {

			if _, ok := dropped[i]; ok {
				continue
			}
			if _, ok := dropped[j]; ok {
				continue
			}

			overlapRatio := calculateOverlapRatio(pointI, boxes[j].Point, "small")

			labelI := boxes[i].Label
			labelJ := boxes[j].Label

			// inline_formula
			if labelI == "inline_formula" || labelJ == "inline_formula" {

				if overlapRatio > 0.5 {
					if labelI == "inline_formula" {
						dropped[i] = struct{}{}
					}
					if labelJ == "inline_formula" {
						dropped[j] = struct{}{}
					}
					continue
				}
			}

			if overlapRatio > 0.7 {

				// polygon 判断
				if layoutShapeMode != "rect" && boxes[i].PolygonPoints != nil {

					polyOverlapRatio := layout.CalculatePolygonOverlapRatio(
						boxes[i].PolygonPoints,
						boxes[j].PolygonPoints,
						"small",
					)

					if polyOverlapRatio < 0.7 {
						continue
					}
				}

				boxAreaI := calculateArea(pointI)
				boxAreaJ := calculateArea(boxes[j].Point)

				// ===== 关键：Python labels 逻辑 =====
				labels := map[string]struct{}{
					labelI: {},
					labelJ: {},
				}

				_, hasImage := labels["image"]
				_, hasTable := labels["table"]
				_, hasSeal := labels["seal"]
				_, hasChart := labels["chart"]

				if (hasImage || hasTable || hasSeal || hasChart) && len(labels) > 1 {

					// Python:
					// if "table" not in labels or labels <= {...}
					if !hasTable || isSubset(labels, map[string]struct{}{
						"table": {},
						"image": {},
						"seal":  {},
						"chart": {},
					}) {
						continue
					}
				}

				if boxAreaI >= boxAreaJ {
					dropped[j] = struct{}{}
				} else {
					dropped[i] = struct{}{}
				}
			}
		}
	}

	// 3️⃣ 过滤
	filtered := make([]*common.LayoutDetResult, 0, len(boxes))
	for i, b := range boxes {
		if _, ok := dropped[i]; !ok {
			filtered = append(filtered, b)
		}
	}

	return filtered
}
func isSubset(a, b map[string]struct{}) bool {
	for k := range a {
		if _, ok := b[k]; !ok {
			return false
		}
	}
	return true
}

// 计算面积
func calculateArea(p []int) float64 {
	if len(p) != 4 {
		return 0
	}
	width := math.Max(0, float64(p[2]-p[0]))
	height := math.Max(0, float64(p[3]-p[1]))
	return width * height
}

// 计算重叠比例
func calculateOverlapRatio(point1 []int, point2 []int, mode string) float64 {

	if len(point1) != 4 || len(point2) != 4 {
		return 0
	}

	// 交集区域
	xMinInter := math.Max(float64(point1[0]), float64(point2[0]))
	yMinInter := math.Max(float64(point1[1]), float64(point2[1]))
	xMaxInter := math.Min(float64(point1[2]), float64(point2[2]))
	yMaxInter := math.Min(float64(point1[3]), float64(point2[3]))

	// 宽高（防止负数）
	interWidth := math.Max(0, xMaxInter-xMinInter)
	interHeight := math.Max(0, yMaxInter-yMinInter)

	interArea := interWidth * interHeight

	// 各自面积
	area1 := calculateArea(point1)
	area2 := calculateArea(point2)

	var refArea float64

	switch mode {
	case "union":
		refArea = area1 + area2 - interArea
	case "small":
		refArea = math.Min(area1, area2)
	case "large":
		refArea = math.Max(area1, area2)
	default:
		return 0
	}
	if refArea == 0 {
		return 0
	}

	return interArea / refArea
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

		return nil, fmt.Errorf("send api error %s  %s", session.Url, err)
	}

	req.Header.Set("Content-Type", "application/json")

	// 6️⃣ 发请求
	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		panic(err)
	}
	defer resp.Body.Close()

	// 7️⃣ 读取返回
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body error %s  %s", session.Url, err)
	}
	var result ChatCompletionResponse
	err = json.Unmarshal(body, &result)

	return &result, nil

}
