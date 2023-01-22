import requests
import csv
import pandas as pd

## Overpass query
# general url
overpass_url = "http://overpass-api.de/api/interpreter?"
# query world
overpass_query = """
    [out:json];
        (node["generator:source"="wind"];);
        out center;
        """


## Send request to overpass API  
code = 504
while(code == 504):
    response = requests.get(overpass_url,
                        params={"data": overpass_query})
    code = response.status_code
datajs = response.json() # save response as json 

# For only reading coordinates
# coords = []
# i = 0 # for testing
# for element in datajs["elements"]:
#     # i = i+1
#     # if i==100: 
#     #     break
#     id = element["id"]
#     if element["type"] == "node":
#         lon = element["lon"]
#         lat = element["lat"]
#         coords.append((lon, lat, id))
#     elif "center" in element: ## get center from polygons 
#         lon = element["center"]["lon"]
#         lat = element["center"]["lat"]
#         coords.append((lon, lat, id))
    

#### Get available tag values of interest from turbines 
coords = []
for element in datajs["elements"]:
    power = "NA"
    height = "NA"
    manufacturer = "NA"
    rotor_size ="NA"
    # i = i+1
    # if i==100: 
    #     break
    if "id" in element:
        if "generator:output:electricity" in element["tags"]:
            if "MW" in element["tags"]["generator:output:electricity"] or "kW" in element["tags"]["generator:output:electricity"]:
                power = element["tags"]["generator:output:electricity"]
                if "," in power:
                    power = power.replace(",", ".")
        if "height:hub" in element["tags"]:
            height = element["tags"]["height:hub"] 
            if "," in height:
                    height = height.replace(",", ".")
        if "manufacturer" in element["tags"]:
            manufacturer = element["tags"]["manufacturer"] 
        if "rotor:diameter" in element["tags"]:
            rotor_size = element["tags"]["rotor:diameter"] 
            if "," in rotor_size:
                    rotor_size = rotor_size.replace(",", ".")
    id = element["id"]
    if element["type"] == "node":
        lon = element["lon"]
        lat = element["lat"]
        coords.append((lon, lat, id, power, height, manufacturer, rotor_size))
    elif "center" in element:
        lon = element["center"]["lon"]
        lat = element["center"]["lat"]
        coords.append((lon, lat, id, power, height, manufacturer, rotor_size))

# Write data to csv file
header = ["lon", "lat", "id", "power", "height", "manufacturer", "rotor_size"]
data = pd.DataFrame(coords)
data.columns = header
data.to_csv("data/data_world_raw.csv")
