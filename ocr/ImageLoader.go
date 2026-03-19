package ocr

import (
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"gocv.io/x/gocv"
)

var httpClient = &http.Client{
	Timeout: 30 * time.Second,
}

func LoadImage(input string) (*gocv.Mat, error) {
	if strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://") {
		return loadFromURL(input)
	}
	return loadFromFile(input)
}

func loadFromFile(path string) (*gocv.Mat, error) {
	img := gocv.IMRead(path, gocv.IMReadColor)
	if img.Empty() {
		return nil, fmt.Errorf("read image failed: %s", path)
	}
	return &img, nil
}

func loadFromURL(url string) (*gocv.Mat, error) {
	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("http get error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("http status error: %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read body error: %w", err)
	}

	img, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		return nil, fmt.Errorf("decode image error: %w", err)
	}

	if img.Empty() {
		return nil, fmt.Errorf("image decode empty")
	}

	return &img, nil
}
