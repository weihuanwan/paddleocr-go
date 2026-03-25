package common

import (
	"fmt"
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
func Sum(nums ...int32) int {
	total := 0
	for _, num := range nums {
		total += int(num)
	}
	return total
}
