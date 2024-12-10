package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/dgravesa/go-parallel/parallel"
)

type pair struct {
	pos   int
	value float32
}

type pairList []pair

func (p pairList) Len() int {
	return len(p)
}

func (p pairList) Less(i, j int) bool {
	return p[i].value < p[j].value
}

func (p pairList) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

const (
	k = 3
	w = 3
	h = 1
)

// Reads an file given the path and returns an float32 array.
func readFile(path string) []float32 {
	file, err := os.Open(path)
	if err != nil {
		fmt.Println("Error: ", err)
		os.Exit(1)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)

	var result []float32

	// Scan the file opened, get the text, remove the spaces and parse to float64.
	for scanner.Scan() {
		x, err := strconv.ParseFloat(strings.TrimSpace(scanner.Text()), 32)
		if err != nil {
			fmt.Println("Error: ", err)
			os.Exit(1)
		}
		// Adds the result to the return array.
		result = append(result, float32(x))
	}

	if err := scanner.Err(); err != nil {
		fmt.Println("Error: ", err)
		os.Exit(1)
	}

	return result
}

// Given the data array, returns the guessing matrix.
func createMatrix(data []float32) [][]float32 {
	n := len(data) - w

	result := make([][]float32, n)

	for i := 0; i < n; i++ {
		result[i] = data[i : i+w]
	}

	return result
}

// Given the xtrain, creates the YTrain array.
func createYTrain(xtrain []float32) []float32 {
	n := len(xtrain) - w

	result := make([]float32, n)

	for i := 0; i < n; i++ {
		result[i] = xtrain[i+w]
	}
	return result
}

// Build the path to the test data.
func buildXTestPath(value string) string {
	var sb strings.Builder
	sb.WriteString("data/dados_xtest_")
	sb.WriteString(value)
	sb.WriteString(".txt")
	return sb.String()
}

// Calculates the euclidian distance between two arrays.
func euclidianDistance(m1 []float32, m2 []float32) float32 {
	var result float64
	for i := 0; i < len(m1); i++ {
		result += math.Pow(float64(m1[i]-m2[i]), 2)
	}
	return float32(result)
}

// Performs the KNN algorithm without parallelism.
func knn(test [][]float32, xtrain [][]float32, ytrain []float32) []float32 {
	result := make([]float32, len(test))

	for result_index, test_row := range test {
		result[result_index] = knnLine(test_row, xtrain, ytrain)
	}

	return result
}

// Performs the KNN algorithm with parallelism.
func parallelLinesKnn(test [][]float32, xtrain [][]float32, ytrain []float32) []float32 {
	result := make([]float32, len(test))

	var wg sync.WaitGroup

	resultChannel := make(chan struct {
		index int
		mean  float32
	}, len(test))

	for result_index, testRow := range test {
		wg.Add(1)

		go func(index int, testRow []float32) {
			defer wg.Done()

			mean := knnLine(testRow, xtrain, ytrain)

			resultChannel <- struct {
				index int
				mean  float32
			}{index, mean}

		}(result_index, testRow)
	}

	go func() {
		wg.Wait()
		close(resultChannel)
	}()

	for res := range resultChannel {
		result[res.index] = res.mean
	}

	return result
}

// Calculates the mean of the k nearest neighbors.
func knnLine(testRow []float32, xtrain [][]float32, ytrain []float32) float32 {
	var mean float32

	all_euclidian_distances := make([]pair, len(xtrain))
	for j, train_row := range xtrain {
		all_euclidian_distances[j] = pair{j, euclidianDistance(testRow, train_row)}
	}

	sort.Sort(pairList(all_euclidian_distances))
	mean = 0
	for index := 0; index < k; index++ {
		mean += ytrain[all_euclidian_distances[index].pos]
	}
	return mean / 3
}

// Performs the KNN algorithm with parallelism using OpenMp.
func parallelOpenMpKnn(test [][]float32, xtrain [][]float32, ytrain []float32) []float32 {
	result := make([]float32, len(test))

	parallel.For(len(test), func(i int, grID int) {
		result[i] = knnLine(test[i], xtrain, ytrain)
	})

	return result
}

func main() {
	values := []string{"10", "30", "50", "100", "1000", "100000", "1000000"}

	xtrain := readFile("data/dados_xtrain.txt")

	xtrain_matrix := createMatrix(xtrain)
	ytrain := createYTrain(xtrain)

	// Running KNN sequential:
	fmt.Println("Running KNN sequential:")
	for _, val := range values {
		testpath := buildXTestPath(val)
		xtest := readFile(testpath)
		xtest_matrix := createMatrix(xtest)

		start := time.Now()

		_ = knn(xtest_matrix, xtrain_matrix, ytrain)

		finish := time.Now()

		fmt.Println(finish.Sub(start))
	}

	// Running KNN parallel:
	fmt.Println("Running KNN parallel:")
	for _, val := range values {
		testpath := buildXTestPath(val)
		xtest := readFile(testpath)
		xtest_matrix := createMatrix(xtest)

		start := time.Now()

		_ = parallelLinesKnn(xtest_matrix, xtrain_matrix, ytrain)

		finish := time.Now()

		fmt.Println(finish.Sub(start))
	}

	// Running KNN parallel with OpenMp:
	fmt.Println("Running KNN parallel OpenMp:")
	for _, val := range values {
		testpath := buildXTestPath(val)
		xtest := readFile(testpath)
		xtest_matrix := createMatrix(xtest)

		start := time.Now()

		_ = parallelOpenMpKnn(xtest_matrix, xtrain_matrix, ytrain)

		finish := time.Now()

		fmt.Println(finish.Sub(start))
	}

	os.Exit(0)
}
