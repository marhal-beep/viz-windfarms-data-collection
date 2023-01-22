import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash import Dash, html
import pandas as pd 

filtered_wt_data = pd.read_csv("data/data_processed_full.csv")[["lon", "lat", "distance"]]
filtered_wt_data =filtered_wt_data[filtered_wt_data.distance<50]

import pandas as pd
import missingno as msno

wf_data= pd.read_csv("data/wf_data_final.csv")
wt_data= pd.read_csv("data/wt_data_final.csv")
plt1 = msno.bar(wf_data, fontsize=18, color="#65BB95")
plt2 = msno.bar(wt_data, fontsize=18, color="#65BB95")

dicts_wt = dlx.geojson_to_geobuf(dlx.dicts_to_geojson(filtered_wt_data.to_dict('records'), lon="lon"))

mapbox_url = "https://api.mapbox.com/styles/v1/mapbox/{id}/tiles/{{z}}/{{x}}/{{y}}{{r}}?access_token={access_token}"
mapbox_token ="pk.eyJ1IjoiendpZWZlbGUiLCJhIjoiY2wxeTAzazJxMDcwaTNibXQ5aTRyMno0bSJ9.zPSVI0wOb_sIXGH-tgHNVw"  # settings.MAPBOX_TOKEN
mapbox_ids = ["light-v9", "dark-v9", "streets-v9", "outdoors-v9", "satellite-streets-v9"]

app = Dash()
app.layout = html.Div([
    dl.Map([#dl.TileLayer(),  
            dl.TileLayer(url=mapbox_url.format(id="satellite-streets-v9", access_token=mapbox_token, noWrap= True)),  
            # dl.TileLayer(),  
            dl.GeoJSON(data = dicts_wt, format = "geobuf", cluster=True, zoomToBoundsOnClick=True, superClusterOptions={"radius": 100, "maxZoom":11}),#,children=[dl.Popup("Displayed Windfarm")]
            dl.GestureHandling(), 
            dl.MeasureControl(position="topleft", primaryLengthUnit="meters", primaryAreaUnit="hectares",
                                    activeColor="#214097", completedColor="#972158"),
            dl.ScaleControl(position="bottomleft")
            

        ], 
    center=(33.256890, -3.810381), 
    zoom=2, 
    preferCanvas=True,
    style={'width': '80vw', 'height': '80vh', 'margin': "auto", "display": "block"})
])

if __name__ == '__main__':
    app.run_server(debug = True)