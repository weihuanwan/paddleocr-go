package common

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"gocv.io/x/gocv"
)

var httpClient = &http.Client{
	Timeout: 30 * time.Second,
}

func LoadImage(input string) (*gocv.Mat, string, error) {
	if strings.HasPrefix(input, "http://") || strings.HasPrefix(input, "https://") {
		return loadFromURL(input)
	}
	return loadFromFile(input)
}

func loadFromFile(path string) (*gocv.Mat, string, error) {
	img := gocv.IMRead(path, gocv.IMReadColor)
	if img.Empty() {
		return nil, "", fmt.Errorf("read image failed: %s", path)
	}

	return &img, getfileName(path), nil
}

func getfileName(imagePath string) string {
	u, err := url.Parse(imagePath)
	if err != nil {
		return ""
	}

	return filepath.Base(u.Path)
}

func loadFromURL(url string) (*gocv.Mat, string, error) {
	resp, err := httpClient.Get(url)
	if err != nil {
		return nil, "", fmt.Errorf("http get error: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, "", fmt.Errorf("http status error: %d", resp.StatusCode)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, "", fmt.Errorf("read body error: %w", err)
	}

	img, err := gocv.IMDecode(data, gocv.IMReadColor)
	if err != nil {
		return nil, "", fmt.Errorf("decode image error: %w", err)
	}

	if img.Empty() {
		return nil, "", fmt.Errorf("image decode empty")
	}

	return &img, getfileName(url), nil
}
