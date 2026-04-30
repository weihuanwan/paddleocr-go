package common

/*
*
int 数组 包含判断
*/
func Contains(arr []int, target int) bool {
	for _, v := range arr {
		if v == target {
			return true
		}
	}
	return false
}
