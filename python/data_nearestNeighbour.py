from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from helpfile import *
from timeit import default_timer as timer
from geopy.distance import geodesic as GD

# Read
data = pd.read_csv("data/data_world_raw.csv")
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis = 1)

# Eckert Projection 
wt_df_eckert = map_projection(data) 
data["lon_proj"] = wt_df_eckert["x"]
data["lat_proj"] = wt_df_eckert["y"]


###testing


# p1 = data.loc[5,["lat", "lon"]].to_list()
# p2 = data.loc[50, ["lat", "lon"]].to_list()
# GD(p1,p2).km


# p1_eck = wt_df_eckert.loc[5,["x", "y"]]
# p2_eck = wt_df_eckert.loc[50, ["x", "y"]]
# math.dist(p1_eck, p2_eck)


# p1_moll = wt_df_mollProj.loc[5,["x", "y"]]
# p2_moll = wt_df_mollProj.loc[50,["x", "y"]]
# math.dist(p1_moll, p2_moll)

def calculate_turbine_spacing(data, maximumDistance):

    # Get data, only look at turbines that have neighbours in 5 km radius 
    dataClustered, dataClusteredGrouped  = dbscan_func(maximumDistance, 2, data, storefile=False)
    dataClustered_gr = dataClustered[dataClustered["WFid"] != -1]

    start = timer()
    wfids = np.unique(dataClustered["WFid"])
    distanceDF = pd.DataFrame(columns = ["id", "closestTurbine", "distance"])
    j = 0
    for wfid in wfids: # For each Wind farm
        currwf = dataClustered_gr[dataClustered_gr["WFid"] == wfid]
        for i in currwf.index: # For each turbine in the wind farm find nearest neighbour and store id and distance in df
            currwt = currwf.loc[i, :]
            badi =currwf.index.isin([i])
            remainingwt = currwf[~badi]
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(remainingwt[["lon_proj", "lat_proj"]].values)
            distance, indices = nbrs.kneighbors(currwt[["lon_proj", "lat_proj"]].values.reshape(1, -1))
            # Fill Data frame
            currId = currwt["id"]
            closesTurbineId = remainingwt.iloc[indices[0][0]]["id"]
            # compute real distance from lon lat location 
            curr_loc = dataClustered_gr.loc[dataClustered_gr["id"] == currId, ["lat", "lon"]].iloc[0].to_list()
            closeturbine_loc = dataClustered_gr.loc[dataClustered_gr["id"] == closesTurbineId, ["lat", "lon"]].iloc[0].to_list()
            dist = GD(curr_loc,closeturbine_loc).m # in meters
            # dist = distance[0][0]
            addDF = pd.DataFrame({"id" :[currId], "closestTurbine":[closesTurbineId], "distance":[dist]})
            distanceDF = pd.concat([distanceDF, addDF])
            j = j+1
            if j%10000 == 0:
                print(j)

    end = timer()
    print(end - start) 

    # distanceDF_full = dataClustered.merge(distanceDF, on ="id")
    return distanceDF

# Whole World 

distanceDF_full = calculate_turbine_spacing(data, 0.5)

# distanceDF_full.to_csv("data/data_processed_withCloseTurbine_new.csv")
distanceDF_full = pd.read_csv("data/data_processed_withCloseTurbine_new.csv")

fig = px.histogram(distanceDF_full["distance"], nbins = 10000)
fig.update_layout(title_text = "Distance of all turbines that have neighbouring turbine within 500 meters")
fig.show()

# Analyze close turbines 
distance_small  = distanceDF_full["distance"][ distanceDF_full["distance"]<30]
fig = px.histogram(distance_small)
fig.show()


## Eliminate turbines that are too close
ids_double = distanceDF_full[distanceDF_full["distance"]<15][["id", "closestTurbine"]]
ids_double["sum_ids"] = ids_double["id"]*ids_double["closestTurbine"]
def f(x): 
    x = list(x)
    return x[0]
ids_to_delete = ids_double.groupby("sum_ids")["id"].agg(f).values
badid = distanceDF_full.loc[distanceDF_full["id"].isin(ids_to_delete)].id.to_list()
goodid = data.loc[~data["id"].isin(badid)].id.to_list()



# goodid.read("data/WT_ids_cleaned.csv")
# goodid = pd.read_csv("data/WT_ids_cleaned.csv")


turbines_distances = calculate_turbine_spacing(data.loc[data["id"].isin(goodid)], 10)
turbines_distances.to_csv("data/data_enrichment/WT_nearestTurbine_distance.csv")