from contextlib import nullcontext
import requests
import pandas as pd
import numpy as np

# Elevation API functions
def getElevationAirMap(lon, lat):
    url = "https://api.airmap.com/elevation/v1/ele/?points=" + str(lat) + "," + str(lon)
    # format query string and return query value
    result = requests.get((url))
    return result.json()["data"][0]

def getElevationOpenElevation(lon, lat):
    url = "https://api.open-elevation.com/api/v1/lookup?locations=" + str(lat) + "," + str(lon)
    # format query string and return query value
    result = requests.get((url))
    return result.json()["results"][0]["elevation"]


data = pd.read_csv("data/data_world_raw.csv")
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis = 1)

## Add Land Coverage data 
wt_lc = pd.read_csv("data/data_enrichment/wt_land_coverage.csv", sep = ";")
wt_lc_names = pd.read_csv("data/data_enrichment/land_coverage_names.csv")
wt_lc_names.columns = ["land_coverage", "land_cover_name"]
wt_landcoverage = wt_lc_names.merge(wt_lc, on = "land_coverage")

# Merge with full dataset
data= data.merge(wt_landcoverage, on = "id", how ="left")

# Add elevations data 
elevations_data = pd.read_csv("data/data_enrichment/wt_elevation.csv",  sep = ";")
elevations_data["elev"] = elevations_data["elev"].replace("#ZAHL!", "NaN")
data = data.merge(elevations_data, on = "id", how ="left")

### Fill in missing data 
# If they are on ocean, set elevation to 0
data["elev"] = np.where((data.elev == "NaN")&(data["land_cover_name"] == "Oceans and seas"), 0, data.elev)

# for the remaining, recomupte through API 
for index, row in data.iterrows():
    if (row["elev"] == "NaN") or (row.elev != row.elev ):
        data.loc[index, "elev"] = getElevationAirMap(row["lon"], row["lat"])

# for the still missing, compute throguh open Elevation API 
for index, row in data.iterrows():
     if (row["elev"] == "NaN") or (row.elev != row.elev ):
        data.loc[index, "elev"] = getElevationOpenElevation(row["lon"], row["lat"])

data_elev = data[["elev"]]
data_elev.to_csv("data/data_enrichment/elevation_api.csv")