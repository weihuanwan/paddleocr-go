package common

import (
	"fmt"
	"image"
	"math"

	"gocv.io/x/gocv"
)

func Round06(x float64) int {
	integer := math.Floor(x)
	frac := x - integer
	if frac >= 0.6 {
		return int(integer + 1)
	}
	return int(integer)
}

// Normalize 对图像进行归一化处理
// resizedImage: 输入图像（gocv.Mat）
// scale: 缩放系数
// mean/std: 均值和标准差，长度为3，分别对应 RGB
func Normalize(resizedImage *gocv.Mat, alpha, beta [3]float32) (*gocv.Mat, error) {
	c := resizedImage.Channels()
	gbrSplit := gocv.Split(*resizedImage)

	if c != 3 {
		return nil, fmt.Errorf("Normalize only supports 3-channel image, got %d", c)
	}

	// 对每个通道做归一化
	for i := 0; i < c; i++ {
		cpMat := gocv.NewMat()
		old := gbrSplit[i]
		err := old.ConvertTo(&cpMat, gocv.MatTypeCV32F)
		if err != nil {
			return nil, fmt.Errorf("Normalize convert to 32F error: %w", err)
		}
		old.Close()
		gbrSplit[i] = cpMat

		gbrSplit[i].MultiplyFloat(alpha[i])
		gbrSplit[i].AddFloat(beta[i])
	}

	result := gocv.NewMat()
	if err := gocv.Merge(gbrSplit, &result); err != nil {
		return nil, fmt.Errorf("Normalize merge error: %w", err)
	}

	return &result, nil
}

/*
*
转换chw
*/
func HWCToCHW(resizedImage *gocv.Mat) []float32 {
	//
	h := resizedImage.Rows()
	w := resizedImage.Cols()
	c := resizedImage.Channels()
	chw := make([]float32, c*h*w)
	data, err := resizedImage.DataPtrFloat32()
	if err != nil {
		panic(err)
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for ch := 0; ch < c; ch++ {
				hwcIndex := (y*w+x)*c + ch
				chwIndex := ch*h*w + y*w + x
				chw[chwIndex] = data[hwcIndex]
			}
		}
	}
	return chw
}
func Resize(imageMat *gocv.Mat, resize [2]int) (*gocv.Mat, []float32, error) {
	resizeMat := gocv.NewMat()

	err := gocv.Resize(*imageMat, &resizeMat, image.Pt(resize[0], resize[1]), 0, 0, gocv.InterpolationCubic)

	if err != nil {
		return nil, nil, fmt.Errorf("wiredTableCells resize failed: %v", err)
	}

	scaleW := float32(resize[0]) / float32(imageMat.Cols())
	scaleH := float32(resize[1]) / float32(imageMat.Rows())

	return &resizeMat, []float32{scaleH, scaleW}, nil
}

func GetNormalizeAlphaBeta() ([3]float32, [3]float32) {
	scale := float32(1.0 / 255.0)
	mean := []float32{0.485, 0.456, 0.406}
	std := []float32{0.229, 0.224, 0.225}
	var alpha [3]float32
	var beta [3]float32

	for i := 0; i < 3; i++ {
		alpha[i] = scale / std[i]
		beta[i] = -mean[i] / std[i]
	}

	return alpha, beta
}
