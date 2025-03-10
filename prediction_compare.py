import plotly.graph_objects as go

categories = [
    "env-1-66-env6-99",
    "env-2-66-env-6-99",
    "env-3-66-env-6-99",
    "env-4-66-env6-99",
    "env-5-66-env-6-99",
    "env-6-66-env-6-99",
    "env-1-99-env6-99",
    "env-2-99-env-6-99",
    "env-3-99-env-6-99",
    "env-4-99-env6-99",
    "env-5-99-env-6-99",
    "env-6-99-env-6-99",
]
values1 = [
    5212500,
    3025000,
    2900000,
    2737500,
    2875000,
    6475000,
    6525000,
    3025000,
    3037500,
    4050000,
    4062500,
    6712500,
]
values2 = [
    5375655.080882094,
    2004903.403194339,
    3244305.541807218,
    3410386.5540429773,
    4351093.902911787,
    6388689.029377501,
    5435570.774198056,
    2418037.449398632,
    3760723.099562584,
    3852850.8662045076,
    4822888.481030554,
    5572395.817389728,
]

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=categories,
        y=values1,
        name="Training Performance",
        marker_color="blue",
        text=values1,
        textposition="outside",
    )
)

fig.add_trace(
    go.Bar(
        x=categories,
        y=values2,
        name="0.25*test_integrals-0.45*control_info+0.26",
        marker_color="red",
        text=values2,
        textposition="outside",
    )
)

fig.update_layout(
    title="Comparison",
    xaxis_title="Categories",
    yaxis_title="Values",
    barmode="group",
    template="plotly_white",
)

fig.show()
