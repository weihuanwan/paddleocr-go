package ocr

import (
	"image"
	"image/color"
	"math"

	"github.com/weihuanwan/paddleocr-go/common"
	ort "github.com/yalue/onnxruntime_go"
	"gocv.io/x/gocv"
)

type RecOnnxSession struct {
	OnnxSession *ort.DynamicAdvancedSession
	Config      *PaddleOCRConfig // 配置信息
	Character   []string         // 字典项
}
type RecResult struct {
	Text  string  // 识别的文本
	Score float32 // 平均置信度
}

func (rec *RecOnnxSession) resizeNormalizeBatch(cropImages []*gocv.Mat) []*gocv.Mat {
	resizedCrops := make([]*gocv.Mat, len(cropImages))
	for i := 0; i < len(cropImages); i++ {
		crop := cropImages[i]
		// 缩放 并且归一化
		resizedCrop := rec.resizeNormalize(crop)
		resizedCrops[i] = resizedCrop
	}
	return resizedCrops
}
func (rec *RecOnnxSession) Run(cropImages []*gocv.Mat) []*RecResult {

	batch := rec.recBatchSize(cropImages)

	batchLen := len(batch)
	result := make([]*RecResult, 0, len(cropImages))
	for i := 0; i < batchLen; i++ {
		crop := batch[i]
		// 缩放+ 归一化
		resizeBatch := rec.resizeNormalizeBatch(crop)
		resLen := len(resizeBatch)

		// 把该一批的图片宽度,填充为 0,最后转换 为 chw
		chwData, h, w := rec.paddedImgHWCToCHW(resizeBatch)
		// 输入描述
		inputShape := ort.NewShape(int64(resLen),
			int64(3),
			int64(h),
			int64(w))
		inputTensor, err := ort.NewTensor(inputShape, chwData)
		if err != nil {
			panic(err)
		}
		// 获取 序列长度（最多字符数） 五舍六入
		seqLen := common.Round06(float64(w) / 8)

		outputTensor, _ := ort.NewEmptyTensor[float32](ort.NewShape(
			int64(resLen),
			int64(seqLen),
			int64(rec.Config.RecModelNumClasses),
		))
		err = rec.OnnxSession.Run([]ort.Value{inputTensor}, []ort.Value{outputTensor})
		if err != nil {
			panic("rec run error")
		}

		texts := rec.CTCDecode(outputTensor.GetData(), resLen, seqLen)
		result = append(result, texts...)
		inputTensor.Destroy()
		outputTensor.Destroy()
	}
	return result
}

func (rec *RecOnnxSession) paddedImgHWCToCHW(batch []*gocv.Mat) ([]float32, int, int) {
	maxW, maxH := maxWH(batch)
	resLen := len(batch)

	allData := make([]float32, resLen*3*maxH*maxW)
	offset := 0
	for i := 0; i < len(batch); i++ {
		img := batch[i]
		maxH = img.Rows()

		newWidth := maxW - img.Cols()
		dst := gocv.NewMat()
		gocv.CopyMakeBorder(
			*img,
			&dst,
			0,        // top
			0,        // bottom
			0,        // left
			newWidth, // right
			gocv.BorderConstant,
			color.RGBA{0, 0, 0, 255}, // padding值
		)
		// 转换chw
		chw := common.HWCToCHW(&dst)
		copy(allData[offset:], chw)
		offset += len(chw)
		dst.Close()
	}
	return allData, maxH, maxW
}

func (rec *RecOnnxSession) Decode(idx []int, prob []float32) (string, float32) {

	var chars []byte
	var scoreSum float32
	count := 0

	last := -1

	for i := 0; i < len(idx); i++ {

		id := idx[i]

		if id == 0 { // blank
			last = id
			continue
		}

		if id == last {
			continue
		}
		chars = append(chars, rec.Character[id]...)
		scoreSum += prob[i]
		count++

		last = id
	}

	if count == 0 {
		return "", 0
	}
	return string(chars), scoreSum / float32(count)
}

func (rec *RecOnnxSession) CTCDecode(
	preds []float32,
	batchSize int,
	seqLen int,
) []*RecResult {

	numClasses := rec.Config.RecModelNumClasses

	texts := make([]*RecResult, batchSize)

	for b := 0; b < batchSize; b++ {
		predsIdx := make([]int, seqLen)
		predsProb := make([]float32, seqLen)

		base := b * seqLen * numClasses

		for t := 0; t < seqLen; t++ {
			/**
			目前字典项是 18385 。
			第一个字符索引：0 - 18384
			第二个字符索引: 18385-(18385 +18384)
			...以此类推下去
			*/
			row := base + t*numClasses

			maxIdx := 0
			maxVal := preds[row]

			for c := 1; c < numClasses; c++ {
				v := preds[row+c]
				if v > maxVal {
					maxVal = v
					maxIdx = c
				}
			}
			predsIdx[t] = maxIdx
			predsProb[t] = maxVal
		}
		text, score := rec.Decode(predsIdx, predsProb)
		texts[b] = &RecResult{Text: text, Score: score}
	}
	return texts
}

func (rec *RecOnnxSession) resizeNormalize(crop *gocv.Mat) *gocv.Mat {

	// 默认
	shape := rec.Config.RecImageShape
	imgH := shape[1]
	imgW := shape[2]
	maxImgW := shape[3]
	maxWHRatio := float64(imgW) / float64(imgH)

	// 检测模型的
	h := crop.Rows() //50
	w := crop.Cols() //481

	defer crop.Close()
	whRatio := float64(w) / float64(h)

	// 取最大的
	maxRatio := max(maxWHRatio, whRatio)

	// 最大的识别宽度
	imgW = int(math.Ceil(float64(imgH) * maxRatio))

	resizedImage := gocv.NewMat()
	var resizedW int

	// 如果大于默认的识别宽度，就使用默认的
	if imgW > maxImgW {
		resizedW = maxImgW
	} else {
		ratio := float64(w) / float64(h)
		temp := int(math.Ceil(float64(imgH) * ratio))
		if temp > imgW {
			resizedW = imgW
		} else {
			resizedW = temp
		}
	}
	err := gocv.Resize(
		*crop,
		&resizedImage,
		image.Pt(resizedW, imgH),
		0,
		0,
		gocv.InterpolationLinear,
	)

	if err != nil {
		panic("缩放失败")
	}

	cpMat := gocv.NewMat()
	alpha := 2.0 / 255.0
	beta := -1.0
	errResizedImage := resizedImage.ConvertToWithParams(&cpMat, gocv.MatTypeCV32FC3, float32(alpha), float32(beta))
	if errResizedImage != nil {
		panic("归一化失败")
	}
	defer resizedImage.Close()
	return &cpMat
}

func (det *RecOnnxSession) recBatchSize(images []*gocv.Mat) [][]*gocv.Mat {
	batchSize := det.Config.RecBatchSize
	var batches [][]*gocv.Mat
	batch := make([]*gocv.Mat, 0, batchSize)
	for _, img := range images {
		batch = append(batch, img)
		if len(batch) == batchSize {
			batches = append(batches, batch)
			batch = nil
		}
	}
	// 最后一批不足 batchSize
	if len(batch) > 0 {
		batches = append(batches, batch)
	}

	return batches
}

func maxWH(imgs []*gocv.Mat) (int, int) {
	maxW := 0
	maxH := 0
	for _, img := range imgs {
		if img.Cols() > maxW {
			maxW = img.Cols()
		}
		maxH = img.Rows()
	}
	return maxW, maxH
}
