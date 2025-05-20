import dash
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import os

DATASET_DIR = "augmented_csv"


def convert_to_float(value):
    try:
        return float(value.replace(",", "."))
    except (ValueError, AttributeError):
        return value

def load_data(file_name):
    file_path = os.path.join(DATASET_DIR, file_name)
    df = pd.read_csv(file_path)
    df = df.loc[:, (df != "Не указано").any()]

    for col in df.columns:
        df[col] = df[col].apply(convert_to_float)

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    columns = df.columns

    task_prefixes = ["A - ", "B - ", "C - ", "D - ", "E - "]
    task_columns = [col for col in df.columns if any(col.startswith(prefix) for prefix in task_prefixes)]
    participant_column = "ФИО"  # Укажите имя колонки с участниками
    score_data = []

    for i, task in enumerate(task_columns):
        task_index = df.columns.get_loc(task)
        next_index = task_index + 1
        
        if i < len(task_columns) - 1:
            next_task_index = df.columns.get_loc(task_columns[i + 1])
            range_columns = df.columns[task_index + 1 : next_task_index]
        else:
            range_columns = df.columns[task_index + 1 :]
        
        task_scores = df[range_columns].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        score_data.append(pd.DataFrame({
            "Участник": df[participant_column],
            "Задача": task[0:2],
            "Баллы": task_scores
        }))

    score_df = pd.concat(score_data, ignore_index=True)
    return score_df

datasets = [f for f in os.listdir(DATASET_DIR) if f.endswith(".csv")]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Div(id="header", className="px-4", children=[
    html.H1("График баллов по задачам", className="my-4"),
    dcc.Dropdown(
        id="dataset-selector",
        options=[{"label": ds, "value": ds} for ds in datasets],
        value=datasets[0],
        style={"width": "100%"}
    ),
    dcc.Graph(id="score-graph")
    ]),
    
])


@app.callback(
    Output("score-graph", "figure"),
    Input("dataset-selector", "value")
)
def update_graph(selected_dataset):
    print(selected_dataset)
    df = load_data(selected_dataset)
    print(df['Баллы'].max())

    fig = px.scatter(
        df,
        x="Задача",
        y="Баллы",
        color="Участник",
        title=f"Распределение баллов по задачам ({selected_dataset})",
        labels={"Баллы": "Сумма баллов", "Задача": "Задачи"},
        symbol="Участник",
        size="Баллы",
    )
    fig.update_layout(
        yaxis=dict(range=[0, df['Баллы'].max()]),
        xaxis_title="Задачи",
        yaxis_title="Баллы",
        height=600
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)