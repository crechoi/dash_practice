import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import dash_table

import pandas as pd
import numpy as np
import plotly.graph_objs as go
import sklearn
from sklearn import datasets
from sklearn.cluster import KMeans
from plotly import tools

app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div([
        html.Div(children="DA", id="second"),
        html.Div(children="AI", id="first"),
        html.Img(src="./assets/dog.gif")
    ], className = "banner"),

    html.Div(id='page-content'),
    dcc.Link('Go back Home!', href = "/", id = "footer")
],className = "home")


@app.callback(Output('page-content', 'children'),[Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/iris_home':
        return iris_layout()
    elif pathname =="/iris_clustering":
        return iris_clustering()
    elif pathname == "/iris_cnn":
        return iris_cnn()
    elif pathname == '/mnist_home':
        return mnist_layout
    else:
        return home_layout


def var_info(col_name, num_array):
    cur_min = np.min(num_array)
    cur_max = np.max(num_array)
    cur_mean = np.mean(num_array)
    return html.Div(children = ["Column Name : {} ".format(col_name), html.Br(),
                                "Min : {} ,".format(cur_min), "Max : {} ,".format(cur_max),
                                "Mean : {}".format(cur_mean), html.Br(), html.Br()])
def box_plotting(df):
    trace0 = go.Box(
    y = df[df.columns[0]],
    name = df.columns[0],
    boxpoints = 'all',
    marker = dict(
        color = 'rgb(7,40,89)'),
    line = dict(
        color = 'rgb(7,40,89)'))

    trace1 = go.Box(
    y = df[df.columns[1]],
    name = df.columns[1],
    boxpoints = 'all',
    marker = dict(
        color = 'rgb(7,140,189)'),
    line = dict(
        color = 'rgb(7,140,189)'))

    trace2 = go.Box(
    y = df[df.columns[2]],
    name = df.columns[2],
    boxpoints = 'all',
    marker = dict(
        color = 'rgb(57,140,89)'),
    line = dict(
        color = 'rgb(57,140,89)'))

    trace3 = go.Box(
    y = df[df.columns[3]],
    name = df.columns[3],
    boxpoints = 'all',
    marker = dict(
        color = 'rgb(107,40,89)'),
    line = dict(
        color = 'rgb(107,40,89)'))

    data = [trace0,trace1,trace2,trace3]
    layout = go.Layout(title = "Box plot for variables", autosize = False, height = 220,
    margin=go.layout.Margin(
        l=20, r=20, b=30, t=30, pad=4),)
    return go.Figure(data, layout)


### Iris CNN
def iris_cnn():
    return html.Div(children=[
            html.Div([

                html.Div(children = [
                                    html.H1("CNN Description"),html.Br()]
                        ),
                html.Img(src = './assets/cnn.png'),
                html.H3("In deep learning, a convolutional neural network is a class of deep neural networks, most commonly applied to analyzing visual imagery....")
                ], id = "content1"),
            html.Div(html.H1("Result of CNN"), id = "content5")
    ], className = "content_main")
### Iris Clustering
def iris_layout():
    df = pd.read_csv('./data/iris.csv')
    return html.Div([
            html.Div(children= [
                html.H2("Raw DATA"),
                dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records')
                )], id = 'content1', style={'overflowY': 'scroll'} ),
            html.Div(children = [
                html.H2("Statistical Information"),
                var_info(df.columns[0], df[df.columns[0]]),
                var_info(df.columns[1],df[df.columns[1]]),
                var_info(df.columns[2],df[df.columns[2]]),
                var_info(df.columns[3],df[df.columns[3]])
            ], id = 'content2'),
            dcc.Graph(figure = box_plotting(df), id = 'content3'),
            html.Div(children = [
                html.H2("Choose model what you want to use"),
                dcc.Link('k-means', href = "iris_clustering"),
                html.Br(),html.Br(),
                dcc.Link('cnn', href = "iris_cnn")
            ], id = 'content4')

        ], className = "content_main")


def iris_clustering():
    df = pd.read_csv('./data/iris.csv')
    return html.Div(children=[
        html.Div(children = [
            html.H1("K-means Algorithm"),
            html.Img(src = "https://miro.medium.com/max/1400/1*dPm3eK6_mYhRTG6ik5YbLQ.png", style = { "width" : "50vh"}),
            html.H3("Clustering Algorithm assigns data into specific number of clusters. Using this algorithm, similar data gather together, others fall apart!")
        ], id = "content1"),
        html.Div(children = [
            html.H1("Result of K-means Algorithm for IRIS"),

            dcc.Dropdown(
                id='my-dropdown',
                options=[
                    {'label': 'K = 3', 'value': '3'},
                    {'label': 'K = 5', 'value': '5'},
                    {'label': 'K = 8', 'value': '8'}
                ],
                value='3'
            ),
            dcc.Graph(id="output-container", style = {"height" : "60vh", "width" : "80vh"})
        ], id = "content5")

    ], className = "content_main")

mnist_layout = html.Div([
    html.Div('mnist_lalyout')
])


@app.callback(Output("output-container", 'figure'),[Input('my-dropdown', 'value')])
def iris_clustering_result(value):
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    k_clustering = KMeans(n_clusters=int(value))
    k_clustering.fit(X)
    labels = k_clustering.labels_
    trace = go.Scatter3d(x=X[:,3], y=X[:,0], z=X[:,2],
                        mode='markers',
                        marker=dict(size =2, color=labels.astype(np.float))
                        )
    data = [trace]
    layout = go.Layout(title = "k-means algorithm")
    return go.Figure(data, layout)



# home layout
home_layout = html.Div([
    html.Div(children = "ML Starts from choosing Data", id = 'description1'),
    html.Br(),html.Br(),
    dcc.Link('MNIST DATA', href='mnist_home'),
    html.Br(),html.Br(),
    dcc.Link('IRIS DATA', href='iris_home')
], className = "main_content")


if __name__ == "__main__":
    app.run_server(debug = False)
