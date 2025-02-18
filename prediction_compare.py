import plotly.graph_objects as go

categories = [
    "env-1-66-env6-99", "env-2-66-env-6-99", "env-3-66-env-6-99", "env-4-66-env6-99", "env-5-66-env-6-99", "env-6-66-env-6-99",
    "env-1-99-env6-99", "env-2-99-env-6-99", "env-3-99-env-6-99", "env-4-99-env6-99", "env-5-99-env-6-99", "env-6-99-env-6-99",
]
values1 = [0.44, 0.42, 0.42, 0.41, 0.26, 0.28, 0.43, 0.38, 0.37, 0.32, 0.28]
values2 = [0.46, 0.40, 0.39, 0.35, 0.26, 0.30,
 0.44, 0.38,0.37, 0.32, 0.29,]

fig = go.Figure()

fig.add_trace(go.Bar(
    x=categories,
    y=values1,
    name="Training Performance",
    marker_color="blue",
    text=values1,
    textposition='outside'
))

fig.add_trace(go.Bar(
    x=categories,
    y=values2,
    name="0.25*test_integrals-0.45*control_info+0.26",
    marker_color="red",
    text=values2,
    textposition='outside'
))

fig.update_layout(
    title="Comparison",
    xaxis_title="Categories",
    yaxis_title="Values",
    barmode="group",
    template="plotly_white"
)

fig.show()
