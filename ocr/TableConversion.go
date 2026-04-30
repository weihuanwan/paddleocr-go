package ocr

import (
	"html"
	"regexp"
	"strconv"
	"strings"
)

/*
*
标记    	含义     	类比
<fcel>	真正单元格	<td>
<ecel>	空单元格		<td></td>
<lcel>	向左合并		colspan
<ucel>	向上合并		rowspan
<xcel>	横+竖合并	rowspan+colspan
<nl>	换行	        <tr>
*/
const (
	OTSL_NL   = "<nl>"   //换行
	OTSL_FCEL = "<fcel>" // 正常单元格
	OTSL_ECEL = "<ecel>" // 空单元格
	OTSL_LCEL = "<lcel>" // 左合并延续
	OTSL_UCEL = "<ucel>" //上合并延续 表示这个格子是“上面那个格子延伸下来的”
	OTSL_XCEL = "<xcel>" // 交叉合并 同时属于横向 + 纵向合并
)

// 每一行的数据
type RowData struct {
	RawCells []string
	TotalLen int // 当前 token 总数
	MinLen   int // 最后一个真实 cell 位置
}

/*
表格列数据
*/
type TableCell struct {
	Text         string
	RowSpan      int
	ColSpan      int
	StartRow     int
	EndRow       int
	StartCol     int
	EndCol       int
	ColumnHeader bool
}

/*
*
表格数据
*/
type TableData struct {
	NumRows    int
	NumCols    int
	TableCells []TableCell
}

/*
*
转化表格的html
*/
func ConvertOtslToHtml(ocrResult string) string {

	newResult := otslPadToSqrV2(ocrResult)
	tokens, textParts := extractTokensAndText(newResult)
	tableCells, splitRowTokens := otslParseTexts(textParts, tokens)

	numRows := len(splitRowTokens)

	numCols := 0
	for _, row := range splitRowTokens {
		if len(row) > numCols {
			numCols = len(row)
		}
	}

	tableData := TableData{
		NumRows:    numRows,
		NumCols:    numCols,
		TableCells: tableCells,
	}
	return exportToHTML(tableData)
}
func (t *TableData) BuildGrid() [][]*TableCell {

	grid := make([][]*TableCell, t.NumRows)

	// 初始化空表
	for i := 0; i < t.NumRows; i++ {

		grid[i] = make([]*TableCell, t.NumCols)

		for j := 0; j < t.NumCols; j++ {
			grid[i][j] = &TableCell{
				Text:     "",
				StartRow: i,
				EndRow:   i + 1,
				StartCol: j,
				EndCol:   j + 1,
			}
		}
	}

	// 覆盖真实 cell
	for _, cell := range t.TableCells {
		for i := min(cell.StartRow, t.NumRows); i < min(cell.EndRow, t.NumRows); i++ {
			for j := min(cell.StartCol, t.NumCols); j < min(cell.EndCol, t.NumCols); j++ {
				grid[i][j] = &cell
			}
		}
	}

	return grid
}

func exportToHTML(tableData TableData) string {

	nrows := tableData.NumRows
	ncols := tableData.NumCols

	if len(tableData.TableCells) == 0 {
		return ""
	}

	grid := tableData.BuildGrid()

	var body strings.Builder

	for i := 0; i < nrows; i++ {

		body.WriteString("<tr>")

		for j := 0; j < ncols; j++ {

			cell := grid[i][j]

			if cell == nil {
				continue
			}

			rowspan := cell.RowSpan
			colspan := cell.ColSpan

			rowstart := cell.StartRow
			colstart := cell.StartCol

			// 👉 不是起点，跳过（关键逻辑）
			if rowstart != i || colstart != j {
				continue
			}

			content := html.EscapeString(
				strings.TrimSpace(cell.Text),
			)

			cellTag := "td"
			if cell.ColumnHeader {
				cellTag = "th"
			}

			var openingTag strings.Builder
			openingTag.WriteString(cellTag)

			if rowspan > 1 {
				openingTag.WriteString(` rowspan="`)
				openingTag.WriteString(intToStr(rowspan))
				openingTag.WriteString(`"`)
			}

			if colspan > 1 {
				openingTag.WriteString(` colspan="`)
				openingTag.WriteString(intToStr(colspan))
				openingTag.WriteString(`"`)
			}

			body.WriteString("<")
			body.WriteString(openingTag.String())
			body.WriteString(">")

			body.WriteString(content)

			body.WriteString("</")
			body.WriteString(cellTag)
			body.WriteString(">")
		}

		body.WriteString("</tr>")
	}

	var result strings.Builder
	result.WriteString("<table border=\"1\">")
	result.WriteString(body.String())
	result.WriteString("</table>")

	return result.String()
}
func intToStr(i int) string {
	return strconv.Itoa(i)
}

func otslParseTexts(texts []string, tokens []string) ([]TableCell, [][]string) {
	// ========= 1️⃣按行拆 =========
	var splitRowTokens [][]string
	var current []string
	for _, t := range tokens {
		if t == OTSL_NL {
			if len(current) > 0 {
				splitRowTokens = append(splitRowTokens, current)
				current = []string{}
			}
		} else {
			current = append(current, t)
		}
	}
	if len(current) > 0 {
		splitRowTokens = append(splitRowTokens, current)
	}

	// ========= 2️⃣补齐矩阵 =========
	if len(splitRowTokens) > 0 {
		// 找到最大的列
		maxCols := 0
		for _, row := range splitRowTokens {
			if len(row) > maxCols {
				maxCols = len(row)
			}
		}

		// 补齐 列
		for i := range splitRowTokens {
			for len(splitRowTokens[i]) < maxCols {
				splitRowTokens[i] = append(splitRowTokens[i], OTSL_ECEL)
			}
		}

		// ========= 3️⃣ 重建 texts =========
		var newTexts []string
		textIdx := 0
		for _, row := range splitRowTokens {
			for _, token := range row {

				newTexts = append(newTexts, token)
				// 判断索引是否越界了
				if textIdx < len(texts) && texts[textIdx] == token {

					textIdx++

					if textIdx < len(texts) && !isTag(texts[textIdx]) {

						newTexts = append(newTexts, texts[textIdx])
						textIdx++
					}
				}
			}

			newTexts = append(newTexts, OTSL_NL)

			if textIdx < len(texts) && texts[textIdx] == OTSL_NL {
				textIdx++
			}
		}

		texts = newTexts
	}

	// ========= 4️⃣ 工具函数 =========
	countRight := func(tokens [][]string, c, r int, targets []string) int {
		span := 0
		for c < len(tokens[r]) && contains(targets, tokens[r][c]) {
			span++
			c++
		}
		return span
	}

	countDown := func(tokens [][]string, c, r int, targets []string) int {
		span := 0
		for r < len(tokens) && contains(targets, tokens[r][c]) {
			span++
			r++
		}
		return span
	}

	// ========= 5️⃣ 主解析 =========
	var tableCells []TableCell
	rIdx := 0
	cIdx := 0

	for i := 0; i < len(texts); i++ {

		text := texts[i]
		cellText := ""

		if text == OTSL_FCEL || text == OTSL_ECEL {

			rowSpan := 1
			colSpan := 1
			rightOffset := 1

			if text != OTSL_ECEL && i+1 < len(texts) {
				cellText = strings.TrimSpace(texts[i+1])
				rightOffset = 2
			}

			var nextRight string
			if i+rightOffset < len(texts) {
				nextRight = texts[i+rightOffset]
			}

			var nextBottom string
			if rIdx+1 < len(splitRowTokens) && cIdx < len(splitRowTokens[rIdx+1]) {
				nextBottom = splitRowTokens[rIdx+1][cIdx]
			}

			// 👉 横向合并
			if nextRight == OTSL_LCEL || nextRight == OTSL_XCEL {
				colSpan += countRight(splitRowTokens, cIdx+1, rIdx,
					[]string{OTSL_LCEL, OTSL_XCEL})
			}

			// 👉 纵向合并
			if nextBottom == OTSL_UCEL || nextBottom == OTSL_XCEL {
				rowSpan += countDown(splitRowTokens, cIdx, rIdx+1,
					[]string{OTSL_UCEL, OTSL_XCEL})
			}

			tableCells = append(tableCells, TableCell{
				Text:     cellText,
				RowSpan:  rowSpan,
				ColSpan:  colSpan,
				StartRow: rIdx,
				EndRow:   rIdx + rowSpan,
				StartCol: cIdx,
				EndCol:   cIdx + colSpan,
			})
		}

		// 👉 移动列
		if isCellToken(text) {
			cIdx++
		}

		// 👉 换行
		if text == OTSL_NL {
			rIdx++
			cIdx = 0
		}
	}

	return tableCells, splitRowTokens

}

func isTag(s string) bool {
	return s == OTSL_NL ||
		s == OTSL_FCEL ||
		s == OTSL_ECEL ||
		s == OTSL_LCEL ||
		s == OTSL_UCEL ||
		s == OTSL_XCEL
}
func isCellToken(s string) bool {
	return s == OTSL_FCEL ||
		s == OTSL_ECEL ||
		s == OTSL_LCEL ||
		s == OTSL_UCEL ||
		s == OTSL_XCEL
}

func contains(arr []string, target string) bool {
	for _, a := range arr {
		if a == target {
			return true
		}
	}
	return false
}

/*
*
完全等价 Python:

def otsl_pad_to_sqr_v2(otsl_str: str) -> str:
*/
func otslPadToSqrV2(otslStr string) string {

	// otsl_str = otsl_str.strip()
	otslStr = strings.TrimSpace(otslStr)

	// if OTSL_NL not in otsl_str:
	if !strings.Contains(otslStr, OTSL_NL) {
		return otslStr + OTSL_NL
	}

	// lines = otsl_str.split(OTSL_NL)
	lines := strings.Split(otslStr, OTSL_NL)

	var rowData []RowData

	for _, line := range lines {

		// if not line:
		if line == "" {
			continue
		}

		// raw_cells = OTSL_FIND_PATTERN.findall(line)
		rawCells := splitOTSL(line)

		// if not raw_cells:
		if len(rawCells) == 0 {
			continue
		}

		// total_len = len(raw_cells)
		totalLen := len(rawCells)

		// min_len = 0
		minLen := 0

		// for i, cell_str in enumerate(raw_cells):
		for i, cellStr := range rawCells {

			// if cell_str.startswith(OTSL_FCEL):
			if strings.HasPrefix(cellStr, OTSL_FCEL) {

				// min_len = i + 1
				minLen = i + 1
			}
		}

		// row_data.append(...)
		rowData = append(rowData, RowData{
			RawCells: rawCells,
			TotalLen: totalLen,
			MinLen:   minLen,
		})
	}

	// if not row_data:
	if len(rowData) == 0 {
		return OTSL_NL
	}

	/*
		global_min_width =
		max(row["min_len"] for row in row_data)
	*/
	globalMinWidth := 0
	maxTotalLen := 0
	for _, row := range rowData {

		globalMinWidth = max(
			globalMinWidth,
			row.MinLen,
		)

		maxTotalLen = max(
			maxTotalLen,
			row.TotalLen,
		)
	}

	// search_start = global_min_width
	searchStart := globalMinWidth

	// search_end = max(global_min_width, max_total_len)
	searchEnd := max(
		globalMinWidth,
		maxTotalLen,
	)

	// min_total_cost = float("inf")
	minTotalCost := int(^uint(0) >> 1)

	// optimal_width = search_end
	optimalWidth := searchEnd

	/*
		找到合适的列数
	*/
	for width := searchStart; width <= searchEnd; width++ {

		currentTotalCost := 0

		for _, row := range rowData {
			currentTotalCost += absInt(row.TotalLen - width)
		}

		if currentTotalCost < minTotalCost {

			minTotalCost = currentTotalCost

			optimalWidth = width
		}
	}

	var repairedLines []string

	for _, row := range rowData {

		// cells = row["raw_cells"]
		cells := row.RawCells

		// current_len = len(cells)
		currentLen := len(cells)

		var newCells []string

		if currentLen > optimalWidth {
			// 如果大于这个列数就截断
			newCells = cells[:optimalWidth]

		} else {
			// 小于列数的就填充上去
			paddingCount := optimalWidth - currentLen

			padding := make([]string, paddingCount)

			for i := 0; i < paddingCount; i++ {
				padding[i] = OTSL_ECEL
			}

			// new_cells = cells + padding
			newCells = append(cells, padding...)
		}

		// 最后得到一个完成的列数表格
		repairedLines = append(
			repairedLines,
			strings.Join(newCells, ""),
		)
	}

	/*
		return OTSL_NL.join(repaired_lines) + OTSL_NL
	*/
	return strings.Join(
		repairedLines,
		OTSL_NL,
	) + OTSL_NL
}

/*
*
替代 Python regex findall
*/
func splitOTSL(line string) []string {

	var result []string

	start := -1

	var tags = []string{
		"<fcel>",
		"<ecel>",
		"<nl>",
		"<lcel>",
		"<ucel>",
		"<xcel>",
	}
	for i := 0; i < len(line); i++ {

		for _, tag := range tags {

			if strings.HasPrefix(line[i:], tag) {

				if start != -1 {

					result = append(
						result,
						line[start:i],
					)
				}

				start = i

				break
			}
		}
	}

	if start != -1 {

		result = append(
			result,
			line[start:],
		)
	}

	return result
}
func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

var tagPattern = regexp.MustCompile(`(<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)`)

func extractTokensAndText(s string) ([]string, []string) {

	// 等价：re.findall
	tokens := tagPattern.FindAllString(s, -1)

	// 等价：re.split（⚠️ 保留分隔符）
	//rawParts := tagPattern.Split(s, -1)

	// Python 的 split + 保留分隔符，其实是：
	// 要把标签也插回去

	matches := tagPattern.FindAllStringIndex(s, -1)

	var textParts []string

	last := 0

	for i, match := range matches {

		start, end := match[0], match[1]

		// 文本部分
		if start > last {
			text := s[last:start]
			if strings.TrimSpace(text) != "" {
				textParts = append(textParts, text)
			}
		}

		// ⚠️ Python split 会把标签也保留下来（因为有括号）
		tag := s[start:end]
		if strings.TrimSpace(tag) != "" {
			textParts = append(textParts, tag)
		}

		last = end

		// 最后一段
		if i == len(matches)-1 && end < len(s) {
			text := s[end:]
			if strings.TrimSpace(text) != "" {
				textParts = append(textParts, text)
			}
		}
	}

	// 如果没有任何匹配
	if len(matches) == 0 {
		if strings.TrimSpace(s) != "" {
			textParts = append(textParts, s)
		}
	}

	return tokens, textParts
}
