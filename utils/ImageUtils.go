package utils

import (
	"context"
	"fmt"
	"image"
	"image/draw"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gen2brain/go-fitz"
	"gocv.io/x/gocv"
)

var (
	defaultHTTPClient = &http.Client{Timeout: 30 * time.Second}
	httpClientMu      sync.RWMutex
)

// SetHTTPClient 设置全局 HTTP 客户端（代理、自定义超时等）
func SetHTTPClient(client *http.Client) {
	httpClientMu.Lock()
	defer httpClientMu.Unlock()
	defaultHTTPClient = client
}

func getHTTPClient() *http.Client {
	httpClientMu.RLock()
	defer httpClientMu.RUnlock()
	return defaultHTTPClient
}

// FileName 从路径或 URL 中提取文件名
func FileName(input string) string {
	if isURL(input) {
		u, err := url.Parse(input)
		if err == nil && u.Path != "" {
			return filepath.Base(u.Path)
		}
		return ""
	}
	return filepath.Base(input)
}

func isURL(input string) bool {
	return strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://")
}

// CloseMats 批量关闭 Mat，通常在 defer 中使用
func CloseMats(mats []gocv.Mat) {
	for i := range mats {
		mats[i].Close()
	}
}

// Load 加载任意输入（本地/URL、图片/PDF），自动识别格式。
// 图片返回单元素切片；PDF 返回所有页切片。
func Load(input string) ([]gocv.Mat, string, error) {
	return LoadWithContext(context.Background(), input)
}

// LoadWithContext 带上下文的 Load
func LoadWithContext(ctx context.Context, input string) ([]gocv.Mat, string, error) {
	name := FileName(input)

	if isURL(input) {
		return loadFromURL(ctx, input, name)
	}
	return loadFromFile(input, name)
}

// -------------------- 本地文件 --------------------

func loadFromFile(path, name string) ([]gocv.Mat, string, error) {
	// 1. 尝试图片
	img := gocv.IMRead(path, gocv.IMReadColor)
	if !img.Empty() {
		return []gocv.Mat{img}, name, nil
	}
	img.Close()

	// 2. 尝试 PDF
	doc, err := fitz.New(path)
	if err != nil {
		return nil, name, fmt.Errorf("unsupported file format: %s", path)
	}
	defer doc.Close()

	mats, err := renderPDFAll(doc)
	if err != nil {
		return nil, name, err
	}
	return mats, name, nil
}

// -------------------- URL --------------------

func loadFromURL(ctx context.Context, urlStr, name string) ([]gocv.Mat, string, error) {
	data, err := downloadBytes(ctx, urlStr)
	if err != nil {
		return nil, name, err
	}

	// 1. 尝试图片
	img, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err == nil && !img.Empty() {
		return []gocv.Mat{img}, name, nil
	}
	if err == nil {
		img.Close()
	}

	// 2. 尝试 PDF：写入临时文件（fitz 需要文件路径）
	tmpFile, err := os.CreateTemp("", "imageutil-*.pdf")
	if err != nil {
		return nil, name, fmt.Errorf("create temp file failed: %w", err)
	}
	tmpPath := tmpFile.Name()

	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		os.Remove(tmpPath)
		return nil, name, fmt.Errorf("write temp file failed: %w", err)
	}
	tmpFile.Close()

	doc, err := fitz.New(tmpPath)
	if err != nil {
		os.Remove(tmpPath)
		return nil, name, fmt.Errorf("unsupported URL content: %s", urlStr)
	}

	mats, loadErr := renderPDFAll(doc)

	doc.Close()
	os.Remove(tmpPath)

	if loadErr != nil {
		return nil, name, loadErr
	}
	return mats, name, nil
}

// -------------------- PDF 渲染 --------------------

func renderPDFPage(doc *fitz.Document, page int) (gocv.Mat, error) {
	if page < 0 || page >= doc.NumPage() {
		return gocv.NewMat(), fmt.Errorf("PDF page out of range: page=%d, total=%d", page, doc.NumPage())
	}
	// ~2480 × 3500
	img, err := doc.ImageDPI(page, 300.0)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("render PDF page %d failed: %w", page+1, err)
	}

	rgbaMat, err := imageToMat(img)
	if err != nil {
		return gocv.NewMat(), fmt.Errorf("convert PDF page %d failed: %w", page+1, err)
	}
	defer rgbaMat.Close()

	bgrMat := gocv.NewMat()
	gocv.CvtColor(rgbaMat, &bgrMat, gocv.ColorRGBAToBGR)
	return bgrMat, nil
}

func renderPDFAll(doc *fitz.Document) ([]gocv.Mat, error) {
	n := doc.NumPage()
	if n == 0 {
		return nil, fmt.Errorf("PDF has no pages")
	}

	mats := make([]gocv.Mat, 0, n)
	for i := 0; i < n; i++ {
		mat, err := renderPDFPage(doc, i)
		if err != nil {
			CloseMats(mats)
			return nil, err
		}
		mats = append(mats, mat)
	}
	return mats, nil
}

// -------------------- 图片转换 --------------------

func imageToMat(img image.Image) (gocv.Mat, error) {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// 快速路径：go-fitz 通常直接返回 *image.RGBA，零拷贝复用
	if rgba, ok := img.(*image.RGBA); ok {
		if rgba.Rect.Min.X == 0 && rgba.Rect.Min.Y == 0 && rgba.Stride == width*4 {
			return gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8UC4, rgba.Pix)
		}
	}

	// 标准路径：draw.Draw 批量拷贝
	rgba := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(rgba, rgba.Bounds(), img, bounds.Min, draw.Src)
	return gocv.NewMatFromBytes(height, width, gocv.MatTypeCV8UC4, rgba.Pix)
}

// -------------------- HTTP 工具 --------------------

func downloadBytes(ctx context.Context, urlStr string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, urlStr, nil)
	if err != nil {
		return nil, fmt.Errorf("create request failed: %w", err)
	}

	resp, err := getHTTPClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("download failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http status error: %d", resp.StatusCode)
	}

	return io.ReadAll(resp.Body)
}
