import numpy as np
import scipy.stats
from typing import List, Tuple, Optional

class Discretizer:
    def __init__(self, ranges: List[Tuple[float, float]], num_buckets: List[int],
                 normal_params: List[Optional[Tuple[float, float]]] = None):
        """
        Initialize the Discretizer.

        :param ranges: List of tuples specifying the min and max value for each dimension. [(min1, max1), (min2, max2), ...]
        :param num_buckets: List of integers specifying the number of buckets for each dimension. [buckets1, buckets2, ...]
                            A value of 0 means no discretization (output the original number),
                            and a value of 1 means all values map to the single bucket midpoint.
        :param normal_params: List of tuples specifying the mean and std for normal distribution for each dimension.
                              If None, use uniform distribution. [(mean1, std1), None, (mean3, std3), ...]
        """
        assert len(ranges) == len(num_buckets), "Ranges and num_buckets must have the same length."
        if normal_params:
            assert len(normal_params) == len(num_buckets), "normal_params must match the length of num_buckets."

        self.ranges: List[Tuple[float, float]] = ranges
        self.num_buckets: List[int] = num_buckets
        self.normal_params: List[Optional[Tuple[float, float]]] = normal_params if normal_params else [None] * len(num_buckets)
        self.bucket_midpoints: List[List[float]] = []

        for i, ((min_val, max_val), buckets, normal_param) in enumerate(zip(ranges, num_buckets, self.normal_params)):
            if buckets > 1:
                if normal_param:
                    mean, std = normal_param
                    # Restrict edges to a finite range if necessary
                    edges = [scipy.stats.norm.ppf(min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std) for j in range(buckets + 1)]
                    midpoints = [round((edges[j] + edges[j + 1]) / 2, 6) for j in range(buckets)]
                else:
                    step = (max_val - min_val) / buckets
                    midpoints = [round(min_val + (i + 0.5) * step, 6) for i in range(buckets)]
                self.bucket_midpoints.append(midpoints)
            else:
                self.bucket_midpoints.append([])

    def discretize(self, vector: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize a vector.

        :param vector: Input vector to discretize. Must have the same length as ranges and num_buckets.
        :return: A tuple of two vectors:
                 - The first vector contains the bucket midpoints (or original value if no discretization).
                 - The second vector contains the bucket indices (or -1 if no discretization).
        """
        assert len(vector) == len(self.ranges), "Input vector must have the same length as ranges."

        midpoints: List[float] = []
        bucket_indices: List[int] = []

        for i, (value, (min_val, max_val), buckets, normal_param) in enumerate(zip(vector, self.ranges, self.num_buckets, self.normal_params)):
            if buckets == 0:
                # No discretization
                midpoints.append(value)
                bucket_indices.append(-1)
            elif buckets == 1:
                # Single bucket, always map to midpoint
                midpoint = round((min_val + max_val) / 2, 6)
                midpoints.append(midpoint)
                bucket_indices.append(0)
            else:
                if normal_param:
                    mean, std = normal_param
                    bucket_edges = [scipy.stats.norm.ppf(min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std) for j in range(buckets + 1)]
                    for idx in range(buckets):
                        if bucket_edges[idx] <= value < bucket_edges[idx + 1]:
                            midpoints.append(round((bucket_edges[idx] + bucket_edges[idx + 1]) / 2, 6))
                            bucket_indices.append(idx)
                            break
                    else:
                        midpoints.append(round((bucket_edges[0] + bucket_edges[-1]) / 2, 6))  # Fallback to average if out of range
                        bucket_indices.append(-1)
                else:
                    step = (max_val - min_val) / buckets
                    bucket = int((value - min_val) / step)
                    bucket = min(max(bucket, 0), buckets - 1)  # Ensure bucket index is within bounds
                    midpoints.append(self.bucket_midpoints[i][bucket])
                    bucket_indices.append(bucket)

        return np.array(midpoints), np.array(bucket_indices)

    def print_buckets(self) -> None:
        """
        Print all buckets and their corresponding ranges.
        """
        for i, ((min_val, max_val), buckets, normal_param) in enumerate(zip(self.ranges, self.num_buckets, self.normal_params)):
            if buckets == 0:
                print(f"Dimension {i}: No discretization")
            elif buckets == 1:
                midpoint = round((min_val + max_val) / 2, 6)
                print(f"Dimension {i}: Single bucket at midpoint {midpoint}")
            else:
                if normal_param:
                    mean, std = normal_param
                    edges = [scipy.stats.norm.ppf(min(max((j / buckets), 1e-6), 1 - 1e-6), loc=mean, scale=std) for j in range(buckets + 1)]
                    for j in range(buckets):
                        bucket_min = round(edges[j], 6)
                        bucket_max = round(edges[j + 1], 6)
                        print(f"Dimension {i}, Bucket {j}: Range [{bucket_min}, {bucket_max})")
                else:
                    step = (max_val - min_val) / buckets
                    for j in range(buckets):
                        bucket_min = round(min_val + j * step, 6)
                        bucket_max = round(bucket_min + step, 6)
                        print(f"Dimension {i}, Bucket {j}: Range [{bucket_min}, {bucket_max})")

if __name__ == "__main__":
    # Define test parameters
    ranges = [(0, 10), (5, 15), (-10, 10), (-np.inf, np.inf)]
    num_buckets = [5, 0, 3, 4]
    normal_params = [None, None, (0, 5), (0, 1)]  # Use normal distribution for the last two dimensions

    # Create Discretizer instance
    discretizer = Discretizer(ranges, num_buckets, normal_params)

    # Print bucket information
    print("Bucket Information:")
    discretizer.print_buckets()

    # Test vectors
    test_vectors = [
        [2, 7, -8, -2],
        [10, 12, 0, 0],
        [5, 15, 10, 2],
        [-1, 5, -5, 1],
        [0, 10, 5, -3]
    ]

    # Apply discretization and print results
    for vector in test_vectors:
        midpoints, indices = discretizer.discretize(vector)
        print(f"\nInput vector: {vector}")
        print(f"Midpoints: {midpoints}")
        print(f"Bucket indices: {indices}")
