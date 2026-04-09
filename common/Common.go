package common

import (
	"fmt"
	"image"
	"math"
	"sort"

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

// SortClockwiseByAngle 将无序点集按顺时针方向排序（图像坐标系）
// 返回一个顺时针排列的顶点切片（假设所有点构成凸多边形）
func SortClockwiseByAngle(points []image.Point) []image.Point {
	if len(points) < 3 {
		return points
	}

	pointsLen := len(points)
	// 读取4个点并计算中心
	quad := make([][2]float64, pointsLen)
	var centerX, centerY float64
	for i := 0; i < pointsLen; i++ {
		quad[i][0] = float64(points[0].X)
		quad[i][1] = float64(points[0].Y)
		centerX += quad[i][0]
		centerY += quad[i][1]
	}
	centerX /= float64(pointsLen)
	centerY /= float64(pointsLen)

	// 按角度排序（逆时针）
	type pointInfo struct {
		idx   int
		angle float64
		x, y  float64
	}

	pts := make([]pointInfo, 4)
	for i := 0; i < 4; i++ {
		pts[i] = pointInfo{
			idx:   i,
			angle: math.Atan2(quad[i][1]-centerY, quad[i][0]-centerX),
			x:     quad[i][0],
			y:     quad[i][1],
		}
	}

	sort.Slice(pts, func(i, j int) bool {
		return pts[i].angle < pts[j].angle
	})

	// 找到左上角（x+y最小）
	topLeftIdx := 0
	minSum := pts[0].x + pts[0].y
	for i := 1; i < 4; i++ {
		if sum := pts[i].x + pts[i].y; sum < minSum {
			minSum = sum
			topLeftIdx = i
		}
	}

	// 构建结果（从左上角开始顺时针）
	result := make([]image.Point, 4)
	for i := 0; i < 4; i++ {
		src := pts[(topLeftIdx+i)%4]
		result[i] = image.Point{
			X: int(math.Round(src.x)),
			Y: int(math.Round(src.y)),
		}
	}

	return result
}
