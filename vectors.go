package vectors

import (
	"math"
	"sort"
)

type Tensor [][]float64

type Callable func(Tensor, Tensor) Tensor

type SearchResult struct {
	CorpusID int
	Score    float64
}

func DotProduct(a, b []float64) float64 {
	result := 0.0
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func Norm(a []float64) float64 {
	result := 0.0
	for _, v := range a {
		result += v * v
	}
	return math.Sqrt(result)
}

func CosSim(queryEmbeddings, corpusEmbeddings Tensor) Tensor {
	numQueries := len(queryEmbeddings)
	numCorpus := len(corpusEmbeddings)
	cosScores := make(Tensor, numQueries)

	for i := 0; i < numQueries; i++ {
		cosScores[i] = make([]float64, numCorpus)
		queryNorm := Norm(queryEmbeddings[i])
		for j := 0; j < numCorpus; j++ {
			cosScores[i][j] = DotProduct(queryEmbeddings[i], corpusEmbeddings[j]) / (queryNorm * Norm(corpusEmbeddings[j]))
		}
	}

	return cosScores
}

func SemanticSearch(queryEmbeddings, corpusEmbeddings Tensor, queryChunkSize, corpusChunkSize, topK int) [][]SearchResult {

	queriesResultList := make([][]SearchResult, len(queryEmbeddings))

	for queryStartIdx := 0; queryStartIdx < len(queryEmbeddings); queryStartIdx += queryChunkSize {
		for corpusStartIdx := 0; corpusStartIdx < len(corpusEmbeddings); corpusStartIdx += corpusChunkSize {
			queryEndIdx := queryStartIdx + queryChunkSize
			if queryEndIdx > len(queryEmbeddings) {
				queryEndIdx = len(queryEmbeddings)
			}

			corpusEndIdx := corpusStartIdx + corpusChunkSize
			if corpusEndIdx > len(corpusEmbeddings) {
				corpusEndIdx = len(corpusEmbeddings)
			}

			cosScores := CosSim(queryEmbeddings[queryStartIdx:queryEndIdx], corpusEmbeddings[corpusStartIdx:corpusEndIdx])

			for queryItr := 0; queryItr < len(cosScores); queryItr++ {
				cosScoresTopKIdx := make([]int, topK)
				cosScoresTopKValues := make([]float64, topK)
				numTopK := 0

				for i := 0; i < len(cosScores[queryItr]); i++ {
					if numTopK < topK || cosScores[queryItr][i] > cosScoresTopKValues[0] {
						insertIdx := 0
						if numTopK < topK {
							insertIdx = numTopK
							numTopK++
						} else {
							insertIdx = 0
						}

						// Shift elements to the right to make space for the new element
						for idx := insertIdx + 1; idx < numTopK; idx++ {
							cosScoresTopKIdx[idx] = cosScoresTopKIdx[idx-1]
							cosScoresTopKValues[idx] = cosScoresTopKValues[idx-1]
						}

						cosScoresTopKIdx[insertIdx] = i
						cosScoresTopKValues[insertIdx] = cosScores[queryItr][i]
					}
				}

				queryID := queryStartIdx + queryItr
				for idx, subCorpusID := range cosScoresTopKIdx {
					corpusID := corpusStartIdx + subCorpusID
					queriesResultList[queryID] = append(queriesResultList[queryID], SearchResult{CorpusID: corpusID, Score: cosScoresTopKValues[idx]})
				}
			}
		}
	}

	for idx := range queriesResultList {
		sort.SliceStable(queriesResultList[idx], func(i, j int) bool {
			return queriesResultList[idx][i].Score > queriesResultList[idx][j].Score
		})
		queriesResultList[idx] = queriesResultList[idx][:topK]
	}

	return queriesResultList
}
