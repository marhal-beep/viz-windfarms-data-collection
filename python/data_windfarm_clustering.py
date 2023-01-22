import pandas as pd
from helpfile import *


# from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'

# Read data
data_clean = pd.read_csv("data/data_processed.csv")#.loc[:,-1]
if 'Unnamed: 0' in data_clean.columns:
    data_clean = data_clean.drop('Unnamed: 0', axis = 1)


# get results from different sizes for epsilon 
for el in [0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 3]:
    res  = dbscan_func(el, 4, data_clean, storefile=False)
    clustered_data,df_grouped = res
    data_clean["WFid_" + str(el)] = clustered_data["WFid"]
   
data_clean.to_csv("data/data_processed_allclusters.csv")


# Create final clustered set 
data = pd.read_csv("data/data_processed.csv")#.loc[:,-1]
data_clustered, t  = dbscan_func(1.3, 2, data, storefile=False)
data_clustered.iloc[:,1::].to_csv("data/data_clustered.csv", index=False)

