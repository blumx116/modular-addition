import pickle

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

with open("all_train_losses.pkl", "rb") as f:
    train_losses: dict[tuple[float, float], list[float]] = pickle.load(f)


with open("all_test_losses.pkl", "rb") as f:
    test_losses: dict[tuple[float, float], list[float]] = pickle.load(f)


def cleaned(
    dictionary: dict[tuple[float, float], list[float]]
) -> dict[tuple[float, float], list[float]]:
    return {
        (round(weight_decay, 2), round(train_frac, 2)): values
        for (weight_decay, train_frac), values in dictionary.items()
    }


train_losses = cleaned(train_losses)
test_losses = cleaned(test_losses)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Graph(id="timeseries-graph"),
        dcc.Slider(
            0,
            1,
            id="weight-decay",
            marks={val: str(val) for val in [0, 0.01, 0.05, 0.1, 0.2]},
            value=0.01,
        ),
        dcc.Slider(
            id="train-frac",
            min=0.05,  # replace with your min value2
            max=1,  # replace with your max value2
            step=0.05,  # replace with your step size
            value=0.7,
        ),
    ]
)


@app.callback(
    Output("timeseries-graph", "figure"),
    [Input("weight-decay", "value"), Input("train-frac", "value")],
)
def update_graph(weight_decay: float, train_frac: float) -> go.Scatter:
    df: pd.DataFrame = pd.DataFrame(
        {
            "train": train_losses[(weight_decay, train_frac)],
            "test": test_losses[(weight_decay, train_frac)],
        }
    )
    return px.line(df)
    # return [
    #     go.Scatter(x=df.index, y=df[column], mode="lines", name=column)
    #     for column in df.columns
    # ]


if __name__ == "__main__":
    app.run_server(debug=True)
