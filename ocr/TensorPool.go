package ocr

import ort "github.com/yalue/onnxruntime_go"

/*
* TODO Tensor 池化 待开发
 */
type TensorPool struct {
	pool chan []ort.Value
}

// 创建一个 TensorPool
func NewTensorPool(size int, shapes [][]int64) *TensorPool {
	pool := make(chan []ort.Value, size)

	for i := 0; i < size; i++ {
		output0, _ := ort.NewEmptyTensor[float32](ort.NewShape(shapes[0]...))
		output1, _ := ort.NewEmptyTensor[int64](ort.NewShape(shapes[1]...))
		output2, _ := ort.NewEmptyTensor[float32](ort.NewShape(shapes[2]...))
		output0.Destroy()
		pool <- []ort.Value{output0, output1, output2}
	}

	return &TensorPool{pool: pool}
}
func (p *TensorPool) Get() []ort.Value {
	select {
	case pool := <-p.pool:
		if pool != nil {
			return pool
		} else {
			return nil
		}
	}

	return nil
}

func (p *TensorPool) Put(tensors []ort.Value) {
	p.pool <- tensors
}
