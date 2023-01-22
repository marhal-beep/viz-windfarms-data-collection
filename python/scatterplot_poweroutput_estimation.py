import plotly.express as px
import pandas as pd
from numpy import corrcoef

# Power vs. Turbine spacing 
data = pd.read_csv("data/full_power_data.csv")
data["col"] = "#2a9d8f"
fig = px.scatter(x=data.distance, y=data.power, color_discrete_sequence=["#2a9d8f"])
fig.update_yaxes(title="power output (MW)")
fig.update_xaxes(title="turbine spacing (m)")
fig.update_layout(title = "power ~ turbine spacing", 
font=dict(
        size=18,
    ))
fig.show()

# Correlation between all variables 
corrcoef(data.distance, data.power)
data.corr()

# Power vs. Rotor size
data = pd.read_csv("data/full_rotor_size_data.csv")
fig = px.scatter(x=data.distance, y=data.rotor_size, color_discrete_sequence=["#2a9d8f"])
fig.update_yaxes(title="rotor size (m)")
fig.update_xaxes(title="turbine spacing (m)")
fig.update_layout(title = "rotor size  ~ turbine spacing",font=dict(
        size=18,
    ))
fig.show()