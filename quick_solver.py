import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import KFold
import plotly.io as pio

# Data
x = np.array([0.15, 0.92, 0.71, 0.70, 0.56, 0.34, 0.89, 0.65, 0.63, 0.45, 0.25, 0.81, 0.55, 0.52, 0.32])
y = np.array([0.00, 0.11, 0.19, 0.20, 0.79, 0.73, 0.10, 0.14, 0.14, 0.31, 0.20, 0.10, 0.11, 0.10, 0.08])
z = np.array([
    4985937, 3662500, 3279687, 3593750, 2692187, 4193750,
    2301562, 1498437, 2992187, 2176562, 5696875,
    1823437, 1693750, 2615625, 2504687,
])
categories = [
    "scratch-env-5-hs",
    "env-1-ls-env5-hs", "env-2-ls-env5-hs", "env-3-ls-env5-hs", "env-4-ls-env5-hs", "env-5-ls-env5-hs",
    "env-1-ms-env5-hs", "env-2-ms-env5-hs", "env-3-ms-env5-hs", "env-4-ms-env5-hs", "env-5-ms-env5-hs",
    "env-1-hs-env5-hs", "env-2-hs-env5-hs", "env-3-hs-env5-hs", "env-4-hs-env5-hs",
]

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

# Compute overall estimation
a, b, c, z_pred = least_squares_fit(x, y, z)

# Create overall estimation figure
fig = go.Figure()
fig.add_trace(go.Bar(
    x=categories,
    y=z,
    name="Training Performance",
    marker_color="blue",
    text=[int(val) for val in z],  # Convert text to integer format
    textposition='outside'
))
fig.add_trace(go.Bar(
    x=categories,
    y=z_pred,
    name="Overall Estimate",
    marker_color="red",
    text=[int(val) for val in z_pred],  # Convert text to integer format
    textposition='outside'
))
fig.update_layout(
    title="Overall Estimation vs Training Performance",
    xaxis_title="Categories",
    yaxis_title="Values",
    barmode="group",
    template="plotly_white"
)

# Save overall estimation plot
pio.write_image(fig, "overall_estimation.png", format="png", scale=1, width=1200, height=800, engine="kaleido")

# Split into 3 groups while keeping the first item in all groups
n_splits = 2
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold = 1

# Ensure first item is always in training data
indices = np.arange(1, len(x))  # Exclude first item (always selected)
np.random.shuffle(indices)
split_size = len(indices) // n_splits
splits = [indices[i * split_size: (i + 1) * split_size] for i in range(n_splits)]

# Assign remaining elements to folds
for i, split in enumerate(splits):
    train_index = np.concatenate(([0], split))  # Ensure first index (0) is always included

    # Select training data
    x_train, y_train, z_train = x[train_index], y[train_index], z[train_index]

    # Fit using the subset (1/3 of the data + first item)
    a_fold, b_fold, c_fold, _ = least_squares_fit(x_train, y_train, z_train)

    # Predict on the full dataset
    z_pred_fold = a_fold * x + b_fold * y + c_fold

    # Create each fold estimation figure
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=z,
        name="Training Performance",
        marker_color="blue",
        text=[int(val) for val in z],  # Convert text to integer format
        textposition='outside'
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=z_pred_fold,
        name=f"Fold {fold} Estimate",
        marker_color="red",
        text=[int(val) for val in z_pred_fold],  # Convert text to integer format
        textposition='outside'
    ))
    fig.add_trace(go.Scatter(
        x=[categories[i] for i in train_index],
        y=[z[i] for i in train_index],
        mode='markers',
        marker=dict(color='green', size=10),
        name=f"Train Data for Fold {fold}"
    ))
    fig.update_layout(
        title=f"Cross-Validation Fold {fold} Estimation",
        xaxis_title="Categories",
        yaxis_title="Values",
        barmode="group",
        template="plotly_white"
    )

    # Save cross-validation plots
    filename = f"cross_validation_fold_{fold}.png"
    pio.write_image(fig, filename, format="png", scale=1, width=1200, height=800, engine="kaleido")

    fold += 1
