from typing import List

from sentence_transformers import util
import random
import time
from torch import Tensor


# Benchmarking (return time)
def benchmark() -> float:
    """
    Benchmarking
    """
    # Start timer
    start = time.time()
    # Generate a list of 10000 random List[float]
    corpus: List[Tensor] = []
    for _ in range(10000):
        # Generate a random List[Tensor] of length 512
        corpus.append(
            Tensor(
                [random.uniform(-1, 1) for _ in range(512)],
            ),
        )
    # Generate 500 queries
    queries: List[Tensor] = []
    for _ in range(500):
        # Generate a random List[Tensor] of length 512
        queries.append(
            Tensor(
                [random.uniform(-1, 1) for _ in range(512)],
            ),
        )
    # Benchmark
    util.semantic_search(
        queries,
        corpus,
    )
    # End timer
    end = time.time()
    total_time = end - start
    # Convert to seconds
    return total_time


if __name__ == "__main__":
    # Benchmark
    print(
        f"Elapsed time for ranking 500 queries against 10000 documents: {benchmark()} seconds"
    )
