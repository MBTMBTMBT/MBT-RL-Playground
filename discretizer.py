import numpy as np
from typing import List, Tuple, Union

class Discretizer:
    def __init__(self, ranges: List[Tuple[float, float]], num_buckets: List[int]):
        """
        Initialize the Discretizer.

        :param ranges: List of tuples specifying the min and max value for each dimension. [(min1, max1), (min2, max2), ...]
        :param num_buckets: List of integers specifying the number of buckets for each dimension. [buckets1, buckets2, ...]
                            A value of 0 means no discretization (output the original number),
                            and a value of 1 means all values map to the single bucket midpoint.
        """
        assert len(ranges) == len(num_buckets), "Ranges and num_buckets must have the same length."

        self.ranges: List[Tuple[float, float]] = ranges
        self.num_buckets: List[int] = num_buckets
        self.bucket_midpoints: List[List[float]] = []

        for (min_val, max_val), buckets in zip(ranges, num_buckets):
            if buckets > 1:
                step = (max_val - min_val) / buckets
                midpoints = [round(min_val + (i + 0.5) * step, 6) for i in range(buckets)]  # Round to 6 decimal places
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

        for i, (value, (min_val, max_val), buckets) in enumerate(zip(vector, self.ranges, self.num_buckets)):
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
                # Regular discretization
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
        for i, ((min_val, max_val), buckets) in enumerate(zip(self.ranges, self.num_buckets)):
            if buckets == 0:
                print(f"Dimension {i}: No discretization")
            elif buckets == 1:
                midpoint = round((min_val + max_val) / 2, 6)
                print(f"Dimension {i}: Single bucket at midpoint {midpoint}")
            else:
                step = (max_val - min_val) / buckets
                for j in range(buckets):
                    bucket_min = round(min_val + j * step, 6)
                    bucket_max = round(bucket_min + step, 6)
                    print(f"Dimension {i}, Bucket {j}: Range [{bucket_min}, {bucket_max})")

# Example Usage
if __name__ == "__main__":
    # Define ranges and number of buckets for each dimension
    ranges = [(0, 10), (-5, 5), (100, 200)]
    num_buckets = [5, 0, 1]

    # Create a Discretizer instance
    discretizer = Discretizer(ranges, num_buckets)

    # Input vector
    input_vector = [7, -3, 150]

    # Discretize the vector
    midpoints, bucket_indices = discretizer.discretize(input_vector)

    print("Midpoints:", midpoints)
    print("Bucket Indices:", bucket_indices)

    # Print all buckets and their ranges
    discretizer.print_buckets()
