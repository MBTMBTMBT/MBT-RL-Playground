import numpy as np


def least_squares_fit(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """
    Solve for a, b, c in the equation ax + by + c = z' using least squares optimization.

    Parameters:
    x (np.ndarray): 1D array of shape (n,)
    y (np.ndarray): 1D array of shape (n,)
    z (np.ndarray): 1D array of shape (n,)

    Returns:
    tuple: (a, b, c, z_pred) where z_pred is the predicted z'
    """
    assert x.shape == y.shape == z.shape, "x, y, and z must have the same shape"

    # Construct the design matrix
    X = np.column_stack((x, y, np.ones_like(x)))  # Shape (n, 3)

    # Solve using least squares method
    theta, _, _, _ = np.linalg.lstsq(X, z, rcond=None)  # theta contains (a, b, c)

    # Compute z' (predicted values)
    z_pred = X @ theta  # Matrix multiplication

    return (*theta, z_pred)


# Example usage
x = np.array([0.14, 0.89, 0.65, 0.63, 0.46, 0.26, 0.25, 0.81, 0.55, 0.52, 0.32, 0.17])
y = np.array([0.00, 0.04, 0.04, 0.045, 0.05, 0.13, 0.05, 0.04, 0.04, 0.035, 0.03, 0.028])
z = np.array([5212500, 3025000, 2900000, 2737500, 2875000, 6475000, 6525000, 3025000, 3037500, 4050000, 4062500, 6712500,])

a, b, c, z_pred = least_squares_fit(x, y, z)
print(f"a = {a}, b = {b}, c = {c}")
print(f"z' = {z_pred.tolist()}")
