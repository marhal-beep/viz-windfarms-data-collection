import random
from random import sample
from plotly.subplots import make_subplots
from helpfile import proj_to_wf_center
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff

wt_data= pd.read_csv("C:/Users/rinar/Documents/Master_Thesis/VizWindfarms_Project/data/wt_data_final.csv")
# wt_data = wt_data[wt_data["Number of turbines"]>3]
# wt_data = wt_data.sort_values("Number of turbines")

wf_data = pd.read_csv("C:/Users/rinar/Documents/Master_Thesis/VizWindfarms_Project/data/wf_data_final.csv")

plot_rows = 14
plot_col = 10

# Draw random sample
wfids = wf_data[wf_data["Number of turbines"]>10].WFid.tolist()
random.seed(32) # set seed for reproducibility in drawing sample 
random_wfs_ids = sample(wfids,plot_rows*plot_col)
random_wfs_ids = wf_data[wf_data["WFid"].isin(random_wfs_ids)].sort_values("Number of turbines").WFid.tolist()

# Create subplot figure and set common map layer
fig = make_subplots(rows=plot_rows, cols=plot_col, horizontal_spacing = 0.01, vertical_spacing = 0.01)
# fig = ff.create_subplots(rows=plot_rows, cols=plot_col, )

fig.update_layout(
    # title = "Random Wind Farms" ,
    showlegend=False, 
    )


x = 0
# Loop over each wind farm and fill row and columns wise 
for i in range(1, plot_rows + 1):
    for j in range(1, plot_col + 1):
        current_wfid = random_wfs_ids[x]
        current_wf = wt_data[wt_data["WFid"] == current_wfid]
        current_wf_proj = proj_to_wf_center(current_wf)
        print(current_wf_proj)
        fig.add_trace(go.Scatter(x = current_wf_proj.loc[:,"x"], y = current_wf_proj.loc[:,"y"], mode='markers',  marker_color='#0B5780'), row=i, col=j)
        x=x+1


fig.update_layout(
    width = 1240, 
    height = 1754, 
    xaxis_title=None, 
    yaxis_title=None
)
# fig.update_layout(plot_bgcolor='lightgray')
fig.update_xaxes(showticklabels=False)
fig.update_yaxes(showticklabels=False)

fig.update_layout(
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=50,
        pad=3
    )
)

fig.show()


