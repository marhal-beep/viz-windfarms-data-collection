import pandas as pd 
import reverse_geocoder as rg
import pycountry
from helpfile import *
import pycountry_convert as pc
pd.options.mode.chained_assignment = None  # default='warn'

data = pd.read_csv("data/data_world_raw.csv")
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis = 1)

columns_raw = data.columns

# Mollweide Projection 
wt_df_mollProj = map_projection(data) 
data["lon_proj"] = wt_df_mollProj["x"]
data["lat_proj"] = wt_df_mollProj["y"]


## Add Country data 
# Get locations for all coordinates 
df_tuples = list(data[["lat", "lon"]].apply(tuple, axis=1))
locs = rg.search(df_tuples)

# Country Information
# Create dict of iso2 key to ountry name value
# iso 3 encoding
countries = {}
for country in pycountry.countries:
    countries[country.alpha_2] = country.name
# Save all locations in list 
country_res = list()
for i in range(len(locs)):
    inst = locs[i]["cc"] # get all locations 
    inst_iso3 = countries.get(inst, 'unknown') # get iso3 of locations code from dict
    country_res.append(inst_iso3)
# Add to data
data["country"] = country_res

# Change turbine of Isle of Man to United Kingdom
data.loc[data["country"] == "Isle of Man", ["Country"]] = "United Kingdom"



# Continent data
def country_name_to_continent_name(country_name):
    if country_name == "unknown":
        return "unknown"
    if country_name == "Antarctica":
        return "Antarctica"
    if country_name == "Western Sahara":
        return "Africa"
    if country_name == "French Southern Territories":
        return "Antarctica"
    else:
        country_alpha2 = pc.country_name_to_country_alpha2(country_name)
        country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
        country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
        return country_continent_name

data.loc[:, "continent"] = data["country"].apply(country_name_to_continent_name)




## Add Land Coverage data 
wt_lc = pd.read_csv("data/data_enrichment/wt_land_coverage.csv", sep = ";")
wt_lc_names = pd.read_csv("data/data_enrichment/land_coverage_names.csv")
wt_lc_names.columns = ["land_coverage", "land_cover_name"]
wt_landcoverage = wt_lc_names.merge(wt_lc, on = "land_coverage")
# Merge with full dataset
data= data.merge(wt_landcoverage, on = "id", how ="left")

# Add elevations data 
elevations_data = pd.read_csv("data/data_enrichment/elevation_api.csv",  sep = ",")[["id", "elev"]]

## Add Land form data
wt_lf = pd.read_csv("data/data_enrichment/wt_landform.csv", sep = ";")
wt_lf_names = pd.read_csv("data/data_enrichment/landform_names.csv")
wt_landforms = wt_lf_names.merge(wt_lf, on = "landform")

# Merge with full datast 
data = data.merge(wt_landforms, on = "id", how ="left")

# Clean ID with nearest turbine > 10m
wt_distance = pd.read_csv("data/data_enrichment/wt_nearestTurbine_distance.csv")[["id", "distance"]]
wt_distance["distance"] = wt_distance["distance"].round(0).apply(int)
cleaned_id = wt_distance.id

# Get rid of too close turbines i.e. double values 
data = data[data["id"].isin(cleaned_id)]
# sum(data.elev == "NaN")



# Add distance to nearest turbine 
data = data.merge(wt_distance, on = "id", how ="left")

# Processed dataset with all columns 
# data.to_csv("data/data_processed_full.csv")
data_reduced = data[[ 'id', 'lon', 'lat', 'lon_proj', 'lat_proj', 'country', 'continent', 'land_cover_name', 'landform_name','elev',  'distance']]
data_reduced.columns = ['id', 'lon', 'lat', 'lon_proj', 'lat_proj', 'Country', 'Continent', 'Land Cover',  'Landform','Elevation', 'Turbine Spacing']



# Fill in land form missing values
# if they are on ocean or permanent water bodies, set land form to "water body"
data_reduced["Landform"] = np.where(data_reduced["Landform"].isnull()&(data_reduced["Land Cover"] == "Oceans and seas"), "Waterbody", data_reduced["Landform"])
data_reduced["Landform"] = np.where(data_reduced["Landform"].isnull()&(data_reduced["Land Cover"] == "Permanent water bodies"), "Waterbody", data_reduced["Landform"])


# Processed data only containing relevant columns
data_reduced.to_csv("data/data_processed.csv")

