package common

import (
	"errors"
	"fmt"
	"github.com/weihuanwan/paddleocr-go/layout"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"math"
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

func CropByBoxes(layoutDet *layout.LayoutDetResult, imageMat gocv.Mat) (gocv.Mat, error) {
	// 参数校验
	if layoutDet == nil {
		return gocv.Mat{}, errors.New("layoutDet is nil")
	}
	if imageMat.Empty() {
		return gocv.Mat{}, errors.New("imageMat is empty")
	}
	if len(layoutDet.Point) != 4 {
		return gocv.Mat{}, errors.New("invalid point format, expected [xmin, ymin, xmax, ymax]")
	}

	xmin, ymin, xmax, ymax := layoutDet.Point[0], layoutDet.Point[1], layoutDet.Point[2], layoutDet.Point[3]

	rect := image.Rect(xmin, ymin, xmax, ymax)
	region := imageMat.Region(rect)

	// 无多边形时直接返回副本（避免原图释放后region失效）
	if len(layoutDet.PolygonPoints) == 0 {
		result := region.Clone()
		return result, nil
	}

	// 创建mask
	mask := gocv.NewMatWithSize(region.Rows(), region.Cols(), gocv.MatTypeCV8U)
	defer mask.Close()

	// 构建局部坐标的多边形
	pts := make([]image.Point, 0, len(layoutDet.PolygonPoints))
	for _, p := range layoutDet.PolygonPoints {
		// 转换为相对于裁剪区域的坐标
		localX := p.X - xmin
		localY := p.Y - ymin

		// 检查转换后的坐标是否在有效范围内
		if localX < 0 || localX >= region.Cols() || localY < 0 || localY >= region.Rows() {
			return gocv.Mat{}, fmt.Errorf("polygon point (%d,%d) out of region bounds after transform", localX, localY)
		}
		pts = append(pts, image.Pt(localX, localY))
	}

	// 填充多边形
	pointsVector := gocv.NewPointsVectorFromPoints([][]image.Point{pts})
	defer pointsVector.Close() // 修复：释放资源

	gocv.FillPoly(&mask, pointsVector, color.RGBA{255, 255, 255, 0})

	// 创建结果图（透明背景更通用，或根据需求改为白色）
	result := gocv.NewMatWithSize(region.Rows(), region.Cols(), region.Type())
	defer func() {
		if err := recover(); err != nil {
			result.Close()
		}
	}()

	// 设置背景色（白色）
	gocv.Rectangle(&result, image.Rect(0, 0, result.Cols(), result.Rows()),
		color.RGBA{255, 255, 255, 0}, -1)

	// 应用mask拷贝有效区域
	region.CopyToWithMask(&result, mask)

	return result, nil
}
