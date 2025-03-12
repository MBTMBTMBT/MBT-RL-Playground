import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import KFold
import plotly.io as pio

# Data
x = np.array(
    [
        0.149,
        0.918,
        0.712,
        0.694,
        0.560,
        0.331,
        0.889,
        0.658,
        0.642,
        0.463,
        0.254,
        0.810,
        0.555,
        0.521,
        0.322,
    ]
)
y = np.array(
    [
        0.00,
        0.002,
        0.005,
        0.006,
        0.030,
        0.028,
        0.002,
        0.003,
        0.003,
        0.008,
        0.005,
        0.002,
        0.002,
        0.002,
        0.002,
    ]
)
z = np.array(
    [
        4834375,
        2315625,
        2521875,
        4125000,
        3956250,
        5275000,
        2793750,
        2246875,
        4721875,
        4153125,
        4800000,
        2853125,
        2806250,
        5146875,
        4665625,
    ]
)
categories = [
    "scratch-env-5-highslipperiness",
    "env-1-lowslipperiness-env5-highslipperiness",
    "env-2-lowslipperiness-env5-highslipperiness",
    "env-3-lowslipperiness-env5-highslipperiness",
    "env-4-lowslipperiness-env5-highslipperiness",
    "env-5-lowslipperiness-env5-highslipperiness",
    "env-1-midlipperiness-env5-highslipperiness",
    "env-2-midlipperiness-env5-highslipperiness",
    "env-3-midlipperiness-env5-highslipperiness",
    "env-4-midlipperiness-env5-highslipperiness",
    "env-5-midlipperiness-env5-highslipperiness",
    "env-1-highslipperiness-env5-highslipperiness",
    "env-2-highslipperiness-env5-highslipperiness",
    "env-3-highslipperiness-env5-highslipperiness",
    "env-4-highslipperiness-env5-highslipperiness",
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

print(a, b, c)

# Create overall estimation figure
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=categories,
        y=z,
        name="Training Performance",
        marker_color="blue",
        text=[int(val) for val in z],  # Convert text to integer format
        textposition="outside",
    )
)
fig.add_trace(
    go.Bar(
        x=categories,
        y=z_pred,
        name="Overall Estimate",
        marker_color="red",
        text=[int(val) for val in z_pred],  # Convert text to integer format
        textposition="outside",
    )
)
fig.update_layout(
    title="Overall Estimation vs Training Performance",
    xaxis_title="Categories",
    yaxis_title="Values",
    barmode="group",
    template="plotly_white",
)

# Save overall estimation plot
pio.write_image(
    fig,
    "overall_estimation.png",
    format="png",
    scale=1,
    width=1200,
    height=800,
    engine="kaleido",
)

# Split into 3 groups while keeping the first item in all groups
n_splits = 2
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
fold = 1

# Ensure first item is always in training data
indices = np.arange(1, len(x))  # Exclude first item (always selected)
np.random.shuffle(indices)
split_size = len(indices) // n_splits
splits = [indices[i * split_size : (i + 1) * split_size] for i in range(n_splits)]

# Assign remaining elements to folds
for i, split in enumerate(splits):
    train_index = np.concatenate(
        ([0], split)
    )  # Ensure first index (0) is always included

    # Select training data
    x_train, y_train, z_train = x[train_index], y[train_index], z[train_index]

    # Fit using the subset (1/3 of the data + first item)
    a_fold, b_fold, c_fold, _ = least_squares_fit(x_train, y_train, z_train)

    # Predict on the full dataset
    z_pred_fold = a_fold * x + b_fold * y + c_fold

    # Create each fold estimation figure
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=z,
            name="Training Performance",
            marker_color="blue",
            text=[int(val) for val in z],  # Convert text to integer format
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=z_pred_fold,
            name=f"Fold {fold} Estimate",
            marker_color="red",
            text=[int(val) for val in z_pred_fold],  # Convert text to integer format
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[categories[i] for i in train_index],
            y=[z[i] for i in train_index],
            mode="markers",
            marker=dict(color="green", size=20),
            name=f"Train Data for Fold {fold}",
        )
    )
    fig.update_layout(
        title=f"Cross-Validation Fold {fold} Estimation",
        xaxis_title="Categories",
        yaxis_title="Values",
        barmode="group",
        template="plotly_white",
    )

    # Save cross-validation plots
    filename = f"cross_validation_fold_{fold}.png"
    pio.write_image(
        fig, filename, format="png", scale=1, width=1200, height=800, engine="kaleido"
    )

    fold += 1
