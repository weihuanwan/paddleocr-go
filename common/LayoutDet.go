package common

import "image"

type LayoutDetBox struct {
	ClsId int
	Label string
	Score float32
	Order int

	Point [4]int

	Mask []int32
}

// 版面分析返回结果
type LayoutDetResult struct {
	ClsId         int           // 标签的 id
	Label         string        // 标签
	Score         float32       // 置信度
	Order         int           // 排序
	Point         []int         // 四边形 4个点位置
	PolygonPoints []image.Point // 多边形位置
}
