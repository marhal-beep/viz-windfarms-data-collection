
from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd
from helpfile import *
import dash_leaflet as dl
from dash.dependencies import Input, Output
import dash_leaflet.express as dlx
from itertools import compress
from sklearn.neighbors import NearestNeighbors
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
# df = readdata("data/clustered_data.csv")
df = pd.read_csv("data/clustered_data.csv")#.iloc[1:5000]
# df = df.drop('Unnamed: 0', axis = 1)

# data_clustered, figures = plotRandomWF(df, "WFid_1.3", l = 2, sample_ids=sample_ids, storeFile=False)
WFids_list = list(df.columns[8:19])


# Create Dropdown
WF_options = [{'label': i, 'value': i} for i in WFids_list]

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Random Wind Farms Spots with different Clustering levels"),
    dcc.Dropdown(id="dd_2", value="WFid_1.2", options=WF_options), 
    html.Div([
        dcc.Graph(id = "choro", style={"display": "inline-block", 'width': '50%'}), 
        dcc.Graph(id = "barplot", style={"display": "inline-block", 'width': '50%'})],
        style={'width': '100%', 'height': '100%' })
])


@app.callback(
    Output(component_id="choro", component_property='figure'), 
    Output(component_id="barplot", component_property='figure'), 
    Input(component_id="dd_2", component_property="value"),
)
def update_plot(value):
    # print(selected_landcover)
    # filtered_wf = [{'lat': i, 'lon': j} for i, j in df[df["land_cover"] == str(value)][["lat", "lon"]].values]
    plot_choro, plot_bar = plot_countryComparison_diffClustering(df, wfid_column = value)

    # positions_new = dlx.dicts_to_geojson(filtered_wf)  
    return  plot_choro, plot_bar



if __name__ == '__main__':
    app.run_server(debug=True, port = "8060")
