import numpy as np
import plotly.graph_objects as go
from scipy.stats import kendalltau

# Data
x = np.array([0.149, 0.918, 0.712, 0.694, 0.560, 0.331, 0.889, 0.658, 0.642, 0.463, 0.254, 0.810, 0.555, 0.521, 0.322])
y = np.array([0.00, 0.002, 0.005, 0.006, 0.030, 0.028, 0.002, 0.003, 0.003, 0.008, 0.005, 0.002, 0.002, 0.002, 0.002])
z = np.array([
    4834375, 2315625, 2521875, 4125000, 3956250, 5275000,
    2793750, 2246875, 4721875, 4153125, 4800000,
    2853125, 2806250, 5146875, 4665625,
])
categories = [
    "scratch-env-5-highslipperiness",
    "env-1-lowslipperiness", "env-2-lowslipperiness",
    "env-3-lowslipperiness", "env-4-lowslipperiness",
    "env-5-lowslipperiness",
    "env-1-midlipperiness", "env-2-midlipperiness",
    "env-3-midlipperiness", "env-4-midlipperiness",
    "env-5-midlipperiness",
    "env-1-highslipperiness", "env-2-highslipperiness",
    "env-3-highslipperiness", "env-4-highslipperiness",
]

# Ensure smaller z gets higher rank
true_rank = np.argsort(z)  # Smallest z gets highest rank

# Generate different values of b in log scale
b_values = np.logspace(-1, 3.5, 2500)  # From 10^-3 to 10^3
ranking_errors = []
ranking_results = []

# Compute rankings for different b values
for b in b_values:
    z_pred = x - b * y
    pred_rank = np.argsort(z_pred)  # Smallest prediction gets highest rank

    # Compute ranking error using Kendall’s Tau
    tau, _ = kendalltau(true_rank, pred_rank)
    ranking_errors.append(1 - tau)  # Error = 1 - correlation

    # Store predicted ranking
    ranking_results.append(pred_rank)

ranking_results = np.array(ranking_results).T

# Preserve the original error curve
fig_error = go.Figure()
fig_error.add_trace(go.Scatter(
    x=b_values,
    y=ranking_errors,
    mode='lines+markers',
    marker=dict(size=6, color='red'),
    name="Ranking Error"
))
fig_error.update_xaxes(type="log")  # Log scale for b
fig_error.update_layout(
    title="Ranking Error vs. b",
    xaxis_title="b (log scale)",
    yaxis_title="Ranking Error (1 - Kendall's Tau)",
    template="plotly_white"
)
fig_error.write_image("ranking_error.png", format="png", scale=1, width=1200, height=800, engine="kaleido")

# Preserve ranking visualization
fig_ranking = go.Figure()
for i, cat in enumerate(categories):
    fig_ranking.add_trace(go.Scatter(
        x=b_values,
        y=ranking_results[i],  # Ranking position
        mode='lines+markers',
        marker=dict(size=8, color=f'rgba({i*15}, {255 - i*15}, {100 + i*10}, 0.7)'),
        name=cat
    ))

fig_ranking.update_xaxes(type="log")  # Log scale for b
fig_ranking.update_yaxes(autorange="reversed")  # Higher rank at the top
fig_ranking.update_layout(
    title="Predicted Ranking vs. b",
    xaxis_title="b (log scale)",
    yaxis_title="Predicted Ranking Position",
    template="plotly_white"
)
fig_ranking.write_image("ranking_visualization.png", format="png", scale=1, width=1200, height=800, engine="kaleido")

# Identify the 3 unique best ranking configurations
unique_ranks = set()
best_b_indices = []

for b_idx in np.argsort(ranking_errors):
    pred_rank_tuple = tuple(ranking_results[:, b_idx])  # Convert ranking array to a tuple for uniqueness
    if pred_rank_tuple not in unique_ranks:
        unique_ranks.add(pred_rank_tuple)
        best_b_indices.append(b_idx)
    if len(best_b_indices) == 3:
        break  # Stop when we have 3 unique rankings

# Generate comparison plots for the best 3 b values
for i, b_idx in enumerate(best_b_indices):
    b_value = b_values[b_idx]
    pred_rank = ranking_results[:, b_idx]  # Corresponding predicted ranking

    # Create scatter plot comparing actual vs predicted ranking
    fig_comparison = go.Figure()

    fig_comparison.add_trace(go.Scatter(
        x=true_rank,
        y=pred_rank,
        mode='markers+text',
        text=categories,  # Show category names
        textposition="top center",  # Adjust text position
        marker=dict(size=10, color="blue"),
        hovertext=categories,  # Ensure text shows on hover
        name=f"Predicted vs Actual Ranking (b={b_value:.4f})"
    ))

    # Add reference line y = x
    fig_comparison.add_trace(go.Scatter(
        x=true_rank,
        y=true_rank,
        mode='lines',
        line=dict(dash='dot', color='red'),
        name="Perfect Match"
    ))

    fig_comparison.update_layout(
        title=f"Best {i+1}: Actual vs Predicted Ranking (b={b_value:.4f})",
        xaxis_title="Actual Ranking",
        yaxis_title="Predicted Ranking",
        template="plotly_white",
        yaxis=dict(autorange="reversed", range=[-1, len(categories)]),  # Ensure text visibility
        xaxis=dict(autorange="reversed", range=[-1, len(categories)])  # Ensure text visibility
    )

    # Save the comparison plot
    filename = f"ranking_comparison_top_{i+1}.png"
    fig_comparison.write_image(filename, format="png", scale=1, width=1200, height=800, engine="kaleido")
