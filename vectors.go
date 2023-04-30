package vectors

import (
	"math"
	"sort"
)

type Tensor [][]float64

type Callable func(Tensor, Tensor) Tensor

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

func SemanticSearch(queryEmbeddings, corpusEmbeddings Tensor, queryChunkSize, corpusChunkSize, topK int) []([]map[string]interface{}) {

	queriesResultList := make([]([]map[string]interface{}), len(queryEmbeddings))

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
				cosScoresTopKIdx := make([]int, 0, topK)
				cosScoresTopKValues := make([]float64, 0, topK)

				for i := 0; i < len(cosScores[queryItr]); i++ {
					if len(cosScoresTopKIdx) < topK || cosScores[queryItr][i] > cosScoresTopKValues[0] {
						cosScoresTopKIdx = append(cosScoresTopKIdx, i)
						cosScoresTopKValues = append(cosScoresTopKValues, cosScores[queryItr][i])

						for idx := len(cosScoresTopKValues) - 1; idx > 0; idx-- {
							if cosScoresTopKValues[idx] > cosScoresTopKValues[idx-1] {
								cosScoresTopKValues[idx], cosScoresTopKValues[idx-1] = cosScoresTopKValues[idx-1], cosScoresTopKValues[idx]
								cosScoresTopKIdx[idx], cosScoresTopKIdx[idx-1] = cosScoresTopKIdx[idx-1], cosScoresTopKIdx[idx]
							} else {
								break
							}
						}

						if len(cosScoresTopKValues) > topK {
							cosScoresTopKValues = cosScoresTopKValues[:topK]
							cosScoresTopKIdx = cosScoresTopKIdx[:topK]
						}
					}
				}

				queryID := queryStartIdx + queryItr
				for idx, subCorpusID := range cosScoresTopKIdx {
					corpusID := corpusStartIdx + subCorpusID
					queriesResultList[queryID] = append(queriesResultList[queryID], map[string]interface{}{"corpus_id": corpusID, "score": cosScoresTopKValues[idx]})
				}
			}
		}
	}

	for idx := range queriesResultList {
		sort.SliceStable(queriesResultList[idx], func(i, j int) bool {
			return queriesResultList[idx][i]["score"].(float64) > queriesResultList[idx][j]["score"].(float64)
		})
		queriesResultList[idx] = queriesResultList[idx][:topK]
	}

	return queriesResultList
}
