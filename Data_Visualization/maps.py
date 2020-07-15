import numpy as np
import pandas as pd
import folium
from folium import plugins

# Define the world map
world_map = folium.Map(location=[56.130, -106.35], zoom_start=4)

# Display world map
world_map.save("map.html")

"""
===============
Stam Toner Maps
===============
"""

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4,
                       tiles="Stamen Toner")
world_map.save("map_stamen_toner.html")

"""
===================
Stamen Terrain Maps
===================
"""

world_map = folium.Map(location=[56.130, -106.35], zoom_start=4,
                       tiles="Stamen Terrain")
world_map.save("map_stamen_terrain.html")

"""
==================
Mapbox Bright Maps
==================
"""

world_map = folium.Map(tiles="Mapbox Bright")
world_map.save("mapbox_bright.html")

"""
=================
Maps with Markers
=================
"""

df_incidents = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Data_Files/Police_Department_Incidents_-_Previous_Year__2016_.csv")

print(df_incidents.head())
print(df_incidents.shape)

# Get the first 100 crimes in the df_incidents dataframe
limit = 100
df_incidents = df_incidents.iloc[0:limit, :]
print(df_incidents.shape)

# San Francisco latitude and longitude values
latitude = 37.77
longitude = -122.42

# Create map and display it
sanfran_map = folium.Map(location=[latitude, longitude], zoom_start=12)
sanfran_map.save("sanfran_map.html")

# Instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

# Loop through the 100 crimes and add each to the incidents feature group
for lat, lng in zip(df_incidents.Y, df_incidents.X):
    incidents.add_child(
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            color="yellow",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6
        )
    )

# Add incidents to map
sanfran_map.add_child(incidents)

# Add pop-up text to each marker on the map
latitudes = list(df_incidents.Y)
longitudes = list(df_incidents.X)
labels = list(df_incidents.Category)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.Marker([lat, lng], popup=label).add_to(sanfran_map)

# Add incidents to map
sanfran_map.add_child(incidents)
sanfran_map.save("sanfran_map.html")

#Marker Cluster

sanfran_map = folium.Map(location = [latitude, longitude],
                         zoom_start = 12)

incidents = plugins.MarkerCluster().add_to(sanfran_map)

# Loop through the dataframe and add each data point to the mark cluster
for lat, lng, label in zip(df_incidents.Y, df_incidents.X,
                           df_incidents.Category):
    folium.Marker(
        location=[lat, lng],
        icon=None,
        popup=label
    ).add_to(incidents)

sanfran_map.save("sanfran_map_cluster.html")

print(df_incidents.Y)


"""
================
Chloropleth Maps
================
"""

df_canada = pd.read_excel('Canada.xlsx',
                       sheet_name='Canada by Citizenship',
                       skiprows=range(20),
                       skipfooter=2)

df_canada.drop(['AREA','REG','DEV','Type','Coverage',],
                  axis=1, inplace=True)

df_canada.rename(columns={'OdName':'Country','AreaName':'Continent',
                       'RegName':'Region'}, inplace=True)

df_canada.columns = list(map(str, df_canada.columns))
df_canada["Total"] = df_canada.sum(axis=1)
years = list(map(str, range(1980,2014)))

# GeoJSON needed to define areas and boundaries
world_geo = r"world_countries.json" # geoJSON file

# Create a numpy array of length 6 and has linear spacing from the
# minimum total immigration to the maximum.
threshold_scale = np.linspace(df_canada["Total"].min(),
                              df_canada["Total"].max(),
                              6, dtype=int)
threshold_scale = threshold_scale.tolist()
threshold_scale[-1] = threshold_scale[-1] + 1 # Make sure that the last
                                              # value of the list is greater
                                              # than min immigration.

# Let Folium determine the scale
world_map = folium.Map(locations=[0, 0], zoom_start=2,
                       tiles="Mapbox Bright")

# Generate chloropleth map using the total immigration of each
# country to Canada from 1980 to 2013
world_map.choropleth(
    geo_data=world_geo,
    data=df_canada,
    columns=["Country", "Total"],
    key_on="feature.properties.name",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Immigration to Canada",
    reset=True

)

world_map.save("choropleth.html")
