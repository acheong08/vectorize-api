package vectors_test

import (
	"encoding/json"
	"math"
	"testing"
	"vectors"

	assert "github.com/stretchr/testify/assert"
)

func TestDotProduct(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}

	expectedResult := 32.0
	actualResult := vectors.DotProduct(a, b)

	assert.Equal(t, expectedResult, actualResult, "Dot product calculation is incorrect")
}

func TestNorm(t *testing.T) {
	a := []float64{1, 2, 3}

	expectedResult := math.Sqrt(14)
	actualResult := vectors.Norm(a)

	assert.Equal(t, expectedResult, actualResult, "Norm calculation is incorrect")
}

func TestCosSim(t *testing.T) {
	queryEmbeddings := vectors.Tensor{
		{1, 0, 0},
		{0, 1, 0},
	}
	corpusEmbeddings := vectors.Tensor{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}

	expectedResult := vectors.Tensor{
		{1, 0, 0},
		{0, 1, 0},
	}
	actualResult := vectors.CosSim(queryEmbeddings, corpusEmbeddings)

	assert.Equal(t, expectedResult, actualResult, "Cosine similarity calculation is incorrect")
}

func TestSemanticSearch(t *testing.T) {
	queryEmbeddings := vectors.Tensor{
		{1, 0, 0},
		{0, 1, 0},
	}
	corpusEmbeddings := vectors.Tensor{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}

	queryChunkSize := 1
	corpusChunkSize := 2
	topK := 2

	expectedResult := [][]vectors.SearchResult{
		{
			{CorpusID: 0, Score: 1},
			{CorpusID: 1, Score: 0},
		}, {
			{CorpusID: 1, Score: 1},
			{CorpusID: 0, Score: 0},
		},
	}
	actualResult := vectors.SemanticSearch(queryEmbeddings, corpusEmbeddings, queryChunkSize, corpusChunkSize, topK)

	// Convert both to JSON string
	expectedJSON, _ := json.Marshal(expectedResult)
	actualJSON, _ := json.Marshal(actualResult)

	assert.Equal(t, expectedJSON, actualJSON, "Semantic search results are incorrect")
}
