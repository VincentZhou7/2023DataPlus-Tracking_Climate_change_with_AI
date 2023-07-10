# %%
#USING 1: SCRIEXTRACTING COO
def extract_coordinates(file_name):
    # Remove the "sentinel_image_" prefix and ".tif" extension from the file name
    file_name = file_name.replace("sentinel_image_", "").replace(".tif", "")
    
    # Split the remaining part by underscore to separate latitude and longitude
    lat, lon = file_name.split("_")
    
    # Convert the latitude and longitude strings to float values
    lat = float(lat)
    lon = float(lon)
    list_coordinates = [lat, lon]
    return list_coordinates

# %%
# first converting method : UTM
import utm

def meters_to_degrees(image_center, image_width_m, image_length_m):

    _, _, zone_number, zone_letter = utm.from_latlon(image_center[0], image_center[1])

    # Convert image center point from UTM to latitude and longitude
    utm_center_x, utm_center_y, _, _ = utm.from_latlon(image_center[0], image_center[1], force_zone_number=zone_number, force_zone_letter=zone_letter)

    half_width_deg = utm.to_latlon(utm_center_x - image_width_m / 2, utm_center_y, zone_number, zone_letter)[1] - image_center[1]
    half_length_deg = image_center[0] - utm.to_latlon(utm_center_x, utm_center_y - image_length_m / 2, zone_number, zone_letter)[0]

    return half_width_deg, half_length_deg


# %%
#2nd converting method: math approximation based on wb84 
import math

def meters_to_degrees(image_center, image_width_m, image_length_m):
    # Calculate the latitude conversion factor at the image center
    lat_conversion_factor = 1 / (111111)

    # Calculate the longitude scaling factor at the image center
    lon_conversion_factor = 1 / (111111)
    # Calculate the half-width and half-length in degrees
    half_width_deg = (image_width_m / 2) * lon_conversion_factor
    half_length_deg = (image_length_m / 2) * lat_conversion_factor

    return half_width_deg, half_length_deg

# %%
import tifffile

def extract_metadata_from_tiff(tiff_file):
    # Load the TIFF file
    tiff = tifffile.TiffFile(tiff_file)

    # Extract metadata
    metadata = {}
    if tiff.pages[0].tags:
        for tag in tiff.pages[0].tags.values():
            tag_name = tag.name
            tag_value = tag.value
            metadata[tag_name] = tag_value

    # Close the TIFF file
    tiff.close()

    return metadata

# %%
def calculate_vertices(image_center, half_length_deg, half_width_deg):
    # Calculate the latitude and longitude values for the four vertices
    topleft = (image_center[0] + half_length_deg, image_center[1] - half_width_deg)
    topright = (image_center[0] + half_length_deg, image_center[1] + half_width_deg)
    bottomright = (image_center[0] - half_length_deg, image_center[1] + half_width_deg)
    bottomleft = (image_center[0] - half_length_deg, image_center[1] - half_width_deg)

    # Arrange the vertices in the desired order (clockwise or counterclockwise)
    vertices = [topleft, topright, bottomright, bottomleft]  # Adjust the order if needed

    return vertices

# %%
#scraping through images
from shapely import Polygon
import os 
part_list = ['part1','part2','part3','part4','part5','part6','part7','part8']
image_data = {}
image_polygon_list = []
image_filename_list =[]

#Exploring TIF Metadata of images
image_folder = '/shared/data/cleaned_images10M/'

geojson_folder = '/shared/data/textual/geojsons'
target_geojson = 'GlobalPowerPlantDatabase.geojson'
geojson_file = os.path.join(geojson_folder, target_geojson)

count = 0

for part in part_list: 
   for filename in os.listdir(image_folder+part):
        center_coordinates = extract_coordinates(filename)
        half_width_deg, half_length_deg = meters_to_degrees(center_coordinates, 224*10, 224*10)
        rectangle_coordinates = calculate_vertices(center_coordinates, half_length_deg, half_width_deg)
        polygon = Polygon(rectangle_coordinates)
        image_data[filename]=polygon
        count+=1
        if (count%50000 == 0): 
                print(count)

# %%
#creating geodataframe
import geopandas as gpd
filenames = list(image_data.keys())
geometries = list(image_data.values())

image_gdf = gpd.GeoDataFrame(geometry=geometries)
image_gdf['filename'] = filenames


# %%
import csv

# Convert the dictionary into a list of tuples
image_list = [(filename, polygon) for filename, polygon in image_data.items()]

csv_file = 'image data testing 2.csv'

# Write the data to CSV
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'geometry'])  # Write header row
    writer.writerows(image_list)  # Write data rows

# %%
import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

df = pd.read_csv("image data testing 2.csv")
df['geometry'] = df['geometry'].apply(lambda x: loads(x))
image_gdf = gpd.GeoDataFrame(df, geometry='geometry')


# %%
#Powerplants 
import os
import pandas as pd
import geopandas as gpd

geojson_folder = '/shared/data/textual/geojsons'
target_geojson = 'GlobalPowerPlantDatabase.geojson'
geojson_file = os.path.join(geojson_folder, target_geojson)
gdf = gpd.read_file(geojson_file)
print(gdf.crs)

# Perform an inner join on the image_gdf and gdf based on spatial intersection
joined_gdf = gpd.sjoin(image_gdf, gdf, how='inner', op='intersects')
joined_df = pd.DataFrame(joined_gdf)
selected_columns = ['name', 'filename', 'geometry']
subset_df = joined_df[selected_columns]
subset_df['Label'] = "Powerplant"

# Print the subset DataFrame
print(subset_df.head(10))

subset_df.to_csv("PowerPlantDatabase.csv")

# %%
subset_df.head(40)

# %%
len(subset_df)

# %%
image_gdf.head()


