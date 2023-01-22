# from re import X
import pandas as pd
import statistics as stats
import numpy as np

def create_data_agg_by_wfid(): 
    wfid_column = "WFid"
    data = pd.read_csv("data/data_clustered.csv")#.iloc[1:5000]
    data[["Turbine Spacing",  "Elevation"]] = data[["Turbine Spacing",  "Elevation"]].round()
    
    # Rename Landform 
    data["Landform"]  = np.where(data["Landform"].isin(["Peak/ridge (cool)", "Peak/ridge", "Peak/ridge (warm)", "Mountain/divide", "Cliff"]), "Summit", data["Landform"])
    data["Landform"]  = np.where(data["Landform"].isin(["Upper slope (warm)", "Upper slope (cool)"]), "Upper slope", data["Landform"])
    data["Landform"]  = np.where(data["Landform"].isin(["Lower slope (warm)",  "Lower slope (cool)", "Lower slope"]), "Lower slope", data["Landform"])
    data["Landform"]  = np.where(data["Landform"].isin(["Valley (narrow)"]), "Valley", data["Landform"])
    data["Landform"]  = np.where(data["Landform"].isin(["Lower slope (flat)", "Upper slope (flat)"]), "Flat", data["Landform"])
    # count nr turbines in park
    wfid_count = data[wfid_column].value_counts()
    wfid_count = wfid_count.to_frame()
    wfid_count.columns = ["Number of turbines"] #change column name

    # take modus value of non numerical columns 
    data_categorical_agg_by_wfid = data[["Country","Continent", "Land Cover", "Landform",wfid_column]].groupby([wfid_column]).agg(lambda x:  stats.mode(x))

    # take mean of numerical columns 
    data_numerical_agg_by_wfid = data[["lon", "lat", "Turbine Spacing",  "Elevation",wfid_column]].groupby([wfid_column]).mean()
    # Merge sets
    windFarms_aggregated_dataset = pd.merge(data_numerical_agg_by_wfid, data_categorical_agg_by_wfid,left_index=True, right_index=True) # merge two datasets 
    windFarms_aggregated_dataset = pd.merge(windFarms_aggregated_dataset, wfid_count,left_index=True, right_index=True) # merge two datasets 

    labels = pd.read_csv("data/data_enrichment/WFshapes_pred.csv")[["WFid", "Shape"]]
    labels = labels.set_index("WFid")

    # Add shape 
    windFarms_aggregated_dataset = windFarms_aggregated_dataset.join(labels)
    windFarms_aggregated_dataset.loc[-1, "Number of turbines"] = 1
    windFarms_aggregated_dataset.loc[windFarms_aggregated_dataset["Shape"].isnull(), ["Shape"]] = "Less than 5 turbines"
    windFarms_aggregated_dataset.loc[-1, "Shape"] = "Single turbine"

    # Rename shapes 
    windFarms_aggregated_dataset["Shape"]  = np.where(windFarms_aggregated_dataset["Shape"].isin([0,4,6,8]), "Lines", windFarms_aggregated_dataset["Shape"])
    windFarms_aggregated_dataset["Shape"]  = np.where(windFarms_aggregated_dataset["Shape"].isin([10]), "Not clustered", windFarms_aggregated_dataset["Shape"])
    windFarms_aggregated_dataset["Shape"]  = np.where(windFarms_aggregated_dataset["Shape"].isin([2]), "Irregular lines", windFarms_aggregated_dataset["Shape"])
    windFarms_aggregated_dataset["Shape"]  = np.where(windFarms_aggregated_dataset["Shape"].isin([5,3]), "Polygon", windFarms_aggregated_dataset["Shape"])
    windFarms_aggregated_dataset["Shape"]  = np.where(windFarms_aggregated_dataset["Shape"].isin([1,7,9]), "Not clustered", windFarms_aggregated_dataset["Shape"])


    # shape_list = np.insert(labels, 0, -10) 
    # len(shape_list)
    # windFarms_aggregated_dataset["Shape"] = shape_list


    #  Roudn numeric values and save in memory saving type 
    windFarms_aggregated_dataset[["Number of turbines"]] = windFarms_aggregated_dataset[["Number of turbines"]].astype("int16")
    windFarms_aggregated_dataset.loc[~windFarms_aggregated_dataset["Elevation"].isnull(), ["Elevation"]] = windFarms_aggregated_dataset.loc[~windFarms_aggregated_dataset["Elevation"].isnull(), ["Elevation"]].astype("int16")
    windFarms_aggregated_dataset.loc[~windFarms_aggregated_dataset["Turbine Spacing"].isnull(), ["Turbine Spacing"]] = windFarms_aggregated_dataset.loc[~windFarms_aggregated_dataset["Turbine Spacing"].isnull(), ["Turbine Spacing"]].astype("int16")

    # Reset column names 
    windFarms_aggregated_dataset = windFarms_aggregated_dataset.reset_index()
    windFarms_aggregated_dataset.columns =  [['WFid', 'lon', 'lat', 'Turbine Spacing', 'Elevation', 'Country', 'Continent', 'Land Cover', 'Landform', 'Number of turbines', 'Shape']] 
    # Write data
    # windFarms_aggregated_dataset.to_csv("data/aggregate_wf_data.csv" ) #write WF data


    # Expand WT data with Number of turbines and Shape 
    turbinescount_df =  pd.DataFrame(windFarms_aggregated_dataset[["WFid", "Number of turbines"]])
    turbinescount_df.columns = ["WFid", "Number of turbines"]
    turbinescount_df.columns  =  turbinescount_df.columns.map(''.join) # get rid of MultiIndex
    wind_turbines_advanced_dataset = data.merge(turbinescount_df, on = "WFid", how = "left")

    # wind_turbines_advanced_dataset = data.merge(wfid_count, on = "WFid",  how ="left")
    shape_df = pd.DataFrame(windFarms_aggregated_dataset[["WFid", "Shape"]])
    turbinescount_df.columns = ["WFid", "Shape"]
    shape_df.columns  =  shape_df.columns.map(''.join) # get rid of MultiIndex
    wind_turbines_advanced_dataset = wind_turbines_advanced_dataset.merge(shape_df, on = "WFid", how = "left")
    
    windFarms_aggregated_dataset[["Land Cover", "Country", "Continent", "Landform"]] = windFarms_aggregated_dataset[["Land Cover", "Country", "Continent", "Landform"]].replace(np.nan, "unknown")
    wind_turbines_advanced_dataset[["Land Cover", "Country", "Continent", "Landform"]] = wind_turbines_advanced_dataset[["Land Cover", "Country", "Continent", "Landform"]].replace(np.nan, "unknown")
    
    windFarms_aggregated_dataset['popup'] = "Displayed"
    wind_turbines_advanced_dataset['popup'] = "Displayed"

    wind_turbines_advanced_dataset = wind_turbines_advanced_dataset[['id', 'lon', 'lat', 'Country', 'Continent','Land Cover', 'Landform', 'Elevation', 'Turbine Spacing', 'WFid', 'Number of turbines', 'Shape', 'popup']]

    # Write data
    windFarms_aggregated_dataset.to_csv("data/wf_data_final.csv", index=False) #write WF data
    wind_turbines_advanced_dataset.to_csv("data/wt_data_final.csv", index=False) #write WF data

    # Write data
    return windFarms_aggregated_dataset, wind_turbines_advanced_dataset
    
a, b = create_data_agg_by_wfid()

idx = a["WFid"].values.tolist()
idx
a[a.WFid.isin(idx)]
# sum(a.Elevation.isnull())
# test = pd.read_csv("data/wt_data_final.csv")
# sum(test.Elevation.isnull())
# b.loc[b["WFid"] != -1, "Turbine Spacing"].median()
# sum(b["Turbine Spacing"].isnull())
# sum(b["Turbine Spacing"].istype(int))
# sum(pd.isna(b["Turbine Spacing"]))
# b["Turbine Spacing"].describe()
# type(b["Turbine Spacing"])
# sum(b["Turbine Spacing"].apply(np.isreal))
# for index, row in b.iterrows():
#     if type(row["Turbine Spacing"]) != int:
#         row["Turbine Spacing"]

# data_windturbines = pd.read_csv("data/wt_data_final.csv")
# data_windturbines[data_windturbines["Elevation"].isnull()]
b["Number of turbines"].max()