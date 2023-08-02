#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

def calculate_vertices(image_center, half_length_deg, half_width_deg):
    # Calculate the latitude and longitude values for the four vertices
    topleft = (image_center[0] + half_length_deg, image_center[1] - half_width_deg)
    topright = (image_center[0] + half_length_deg, image_center[1] + half_width_deg)
    bottomright = (image_center[0] - half_length_deg, image_center[1] + half_width_deg)
    bottomleft = (image_center[0] - half_length_deg, image_center[1] - half_width_deg)

    # Arrange the vertices in the desired order (clockwise or counterclockwise)
    vertices = [topleft, topright, bottomright, bottomleft]  # Adjust the order if needed

    return vertices


# In[10]:


"""
NOT USING THIS METHOD OF PROJECTING COORDINATES, ALTHOUGH IT IS RELEVANT

import utm

def meters_to_degrees(image_center, image_width_m, image_length_m):
    utm_center_x, utm_center_y, zone_number, zone_letter = utm.from_latlon(image_center[0], image_center[1])
    northern_hemisphere = image_center[0] >= 0

    if northern_hemisphere:
        half_width_deg = utm.to_latlon(utm_center_x - image_width_m / 2, utm_center_y, zone_number, zone_letter)[1] - image_center[1]
        half_length_deg = utm.to_latlon(utm_center_x, utm_center_y + image_length_m / 2, zone_number, zone_letter)[0] - image_center[0]
    else:
        half_width_deg = image_center[1] - utm.to_latlon(utm_center_x - image_width_m / 2, utm_center_y, zone_number, zone_letter)[1]
        half_length_deg = image_center[0] - utm.to_latlon(utm_center_x, utm_center_y - image_length_m / 2, zone_number, zone_letter)[0]

    return half_width_deg, half_length_deg

"""


# In[15]:


import math

def meters_to_degrees(image_center, image_width_m, image_length_m):
    latitude = image_center[0]

    # Calculate the number of meters in one degree of latitude at the given latitude
    meters_per_degree_lat = 111319.9 
    # Calculate the number of meters in one degree of longitude at the given latitude
    meters_per_degree_lon = 111319.9
    # Convert the image width and length to degrees
    half_width_deg = (image_width_m / meters_per_degree_lon) / 2
    half_length_deg = (image_length_m / meters_per_degree_lat) / 2

    return half_width_deg, half_length_deg


# In[12]:


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


# In[15]:


extract_metadata_from_tiff("/shared/data/cleaned_images10M/part1/sentinel_image_56.20172700864569_-3.7893484760238656.tif")


# In[16]:


from shapely import Polygon
import time
import os 
part_list = ['part1','part2','part3','part4','part5','part6','part7','part8']
image_data = {}
image_polygon_list = []
image_filename_list =[]

#Exploring TIF Metadata of images
image_folder = '/shared/data/cleaned_images10M/'
count = 0
for part in part_list:
   for filename in os.listdir(image_folder+part):
        center_coordinates = extract_coordinates(filename)
        half_width_deg, half_length_deg = meters_to_degrees(center_coordinates, 224*10, 224*10)
        rectangle_coordinates = calculate_vertices(center_coordinates, half_length_deg, half_width_deg)
        polygon = Polygon(rectangle_coordinates)
        image_data["/shared/data/cleaned_images10M/"+part+'/'+filename]=polygon
        count+=1
        if (count%50000 == 0): 
                print(count)  
   print(part+"complete")               


# In[17]:


#creating geodataframe
import geopandas as gpd
filenames = list(image_data.keys())
geometries = list(image_data.values())

image_gdf = gpd.GeoDataFrame(geometry=geometries)
image_gdf['filename'] = filenames


# In[5]:


import csv

# Convert the dictionary into a list of tuples
image_list = [(filename, polygon) for filename, polygon in image_data.items()]

csv_file = 'Image Data Math.csv'

# Write the data to CSV
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'geometry'])  # Write header row
    writer.writerows(image_list)  # Write data rows


# In[1]:


#method to convert geojson to merged csv 
import os
import pandas as pd
import geopandas as gpd

def geojson_to_csv(desired_file, name, label, naming_var):
        geojson_folder = '/shared/data/textual/geojsons'
        target_geojson = desired_file
        geojson_file = os.path.join(geojson_folder, target_geojson)
        gdf = gpd.read_file(geojson_file)
        print(gdf.crs)

        # Perform an inner join on the image_gdf and gdf based on spatial intersection
        joined_gdf = gpd.sjoin(image_gdf, gdf, how='inner', op='intersects')
        joined_df = pd.DataFrame(joined_gdf)
        selected_columns = [name, 'filename', 'geometry']
        subset_df = joined_df[selected_columns]
        subset_df['label'] = label

        # Print the subset DataFrame
        print(subset_df.head(2))

        subset_df.to_csv(naming_var+"Database.csv")


# In[9]:


import pandas as pd
import geopandas as gpd
from shapely.wkt import loads

# Read the CSV file
df = pd.read_csv("Image Data Math.csv")

# Convert the 'geometry' column from WKT to actual geometries
df['geometry'] = df['geometry'].apply(loads)

# Create the GeoDataFrame
image_gdf = gpd.GeoDataFrame(df, geometry='geometry')
len(image_gdf)


# In[11]:


geojson_to_csv("GlobalPowerPlantDatabase.geojson",'name','PowerPlant','PowerPlant')
geojson_to_csv("GlobalAirportDatabase.geojson","Airport Name",'Airport','Airport')
geojson_to_csv("GlobalCoalandMetalMining.geojson","mine_fac",'Mine','Mine')
geojson_to_csv("OffshoreInstallations.geojson",'Name',"OffshoreInstallation","OffshoreInstallation")
geojson_to_csv("USWindTurbineDatabase.geojson","Name",'Wind Turbine','Wind Turbine')
geojson_to_csv("cropHarvest.geojson","Label",'CropHarvest','CropHarvest')
geojson_to_csv("WorldPortIndex.geojson","Main Port Name",'Port','Port')
geojson_to_csv("NaturalEarth_Roads.geojson","Country",'Road','Road1')
geojson_to_csv("GlaciersElevationMass.geojson","Name",'Glacier','Glacier')






# In[14]:


geojson_to_csv("glims_polygons.geojson","Glacier Name",'Glacier','Glacier2')


# In[ ]:


geojson_to_csv("GRIP4_Region1.geojson","RoadCountry",'Road','Road4')



# In[ ]:


geojson_to_csv("GRIP4_region1geojson","RoadCountry",'Road','Road3')


# In[ ]:


geojson_to_csv("GRIP4_region7.geojson","RoadCountry",'Road','Road2')


# In[31]:


pip install --upgrade scikit-learn numpy


# In[35]:


pip install --upgrade scikit-learn numpy


# In[36]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Splitting 70% for training, and 30% for test + validate
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)

# Splitting the remaining 30% into 20% for testing and 10% for validation
test_df, validate_df = train_test_split(temp_df, test_size=0.33, random_state=42)


# In[37]:


pip install matplotlib pillow


# In[1]:


import matplotlib.pyplot as plt
from PIL import Image

# Replace the file paths with the actual paths of your TIF images
tif_files = [
    "/shared/data/cleaned_images10M/part1/sentinel_image_41.68908491822264_-87.41747086196023.tif",
    "/shared/data/cleaned_images10M/part2/sentinel_image_41.68908491822264_-87.41747086196023.tif",
    "/shared/data/cleaned_images10M/part3/sentinel_image_41.68908491822264_-87.41747086196023.tif",
    "/shared/data/cleaned_images10M/part4/sentinel_image_41.68908491822264_-87.41747086196023.tif"
]

# Create a subplot to display multiple images
num_images = len(tif_files)
num_cols = 2
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and iterate through the images to display
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        tif_image = Image.open(tif_files[i])
        ax.imshow(tif_image)
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty subplots if there are fewer images than subplots

plt.tight_layout()  # Adjust the layout for better visualization
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from PIL import Image

# Replace the file paths with the actual paths of your TIF images
tif_files = [
    "/shared/data/cleaned_images10M/part1/sentinel_image_41.68908491822264_-87.41747086196023.tif",
    "/shared/data/cleaned_images10M/part2/sentinel_image_41.68908491822264_-87.41747086196023.tif",
    "/shared/data/cleaned_images10M/part3/sentinel_image_41.68908491822264_-87.41747086196023.tif",
    "/shared/data/cleaned_images10M/part4/sentinel_image_41.68908491822264_-87.41747086196023.tif"
]

# Create a subplot to display multiple images
num_images = len(tif_files)
num_cols = 2
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and iterate through the images to display
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        tif_image = Image.open(tif_files[i])
        ax.imshow(tif_image)
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty subplots if there are fewer images than subplots

plt.tight_layout()  # Adjust the layout for better visualization
plt.show()


# In[41]:


import matplotlib.pyplot as plt
from PIL import Image

# Replace the file paths with the actual paths of your TIF images
tif_files = [
    "/shared/data/cleaned_images10M/part1/sentinel_image_51.2441717295636_3.2131988267446285.tif",
    "/shared/data/cleaned_images10M/part2/sentinel_image_51.2441717295636_3.2131988267446285.tif",
    "/shared/data/cleaned_images10M/part3/sentinel_image_51.2441717295636_3.2131988267446285.tif",
    "/shared/data/cleaned_images10M/part4/sentinel_image_51.2441717295636_3.2131988267446285.tif",
    "/shared/data/cleaned_images10M/part2/sentinel_image_51.2441717295636_3.2131988267446285.tif"
]

# Create a subplot to display multiple images
num_images = len(tif_files)
num_cols = 3
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and iterate through the images to display
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        tif_image = Image.open(tif_files[i])
        ax.imshow(tif_image)
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty subplots if there are fewer images than subplots

plt.tight_layout()  # Adjust the layout for better visualization
plt.show()


# In[43]:


import matplotlib.pyplot as plt
from PIL import Image

# Replace the file paths with the actual paths of your TIF images
tif_files = [
    "/shared/data/cleaned_images10M/part1/sentinel_image_-16.503290840296202_-68.20071030407543.tif",
    "/shared/data/cleaned_images10M/part3/sentinel_image_-16.503290840296202_-68.20071030407543.tif",
    "/shared/data/cleaned_images10M/part4/sentinel_image_-16.503290840296202_-68.20071030407543.tif",
    "/shared/data/cleaned_images10M/part5/sentinel_image_-16.503290840296202_-68.20071030407543.tif",
    "/shared/data/cleaned_images10M/part8/sentinel_image_-16.503290840296202_-68.20071030407543.tif"
]

# Create a subplot to display multiple images
num_images = len(tif_files)
num_cols = 3
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and iterate through the images to display
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        tif_image = Image.open(tif_files[i])
        ax.imshow(tif_image)
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty subplots if there are fewer images than subplots

plt.tight_layout()  # Adjust the layout for better visualization
plt.show()


# In[46]:


import matplotlib.pyplot as plt
from PIL import Image

# Replace the file paths with the actual paths of your TIF images
tif_files = [
    "/shared/data/cleaned_images10M/part3/sentinel_image_18.490139198891924_78.10825934658466.tif",
    "/shared/data/cleaned_images10M/part4/sentinel_image_18.490139198891924_78.10825934658466.tif",
    "/shared/data/cleaned_images10M/part8/sentinel_image_18.490139198891924_78.10825934658466.tif",
 
]

# Create a subplot to display multiple images
num_images = len(tif_files)
num_cols = 3
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and iterate through the images to display
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        tif_image = Image.open(tif_files[i])
        ax.imshow(tif_image)
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty subplots if there are fewer images than subplots

plt.tight_layout()  # Adjust the layout for better visualization
plt.show()


# In[11]:


import matplotlib.pyplot as plt
from PIL import Image

# Replace the file paths with the actual paths of your TIF images
tif_files = [
    "/shared/data/cleaned_images10M/part2/sentinel_image_51.25676659728471_3.2131988267446285.tif",
    "/shared/data/cleaned_images10M/part2/sentinel_image_51.2441717295636_3.2131988267446285.tif"
]

# Create a subplot to display multiple images
num_images = len(tif_files)
num_cols = 1
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and iterate through the images to display
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        tif_image = Image.open(tif_files[i])
        ax.imshow(tif_image)
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty subplots if there are fewer images than subplots

plt.tight_layout()  # Adjust the layout for better visualization
plt.show()


# In[4]:



import matplotlib.pyplot as plt
from PIL import Image

# Replace the file paths with the actual paths of your TIF images
tif_files = [
    "/shared/data/cleaned_images10M/part6/sentinel_image_40.91816875451951_-72.74834159840209.tif",
"/shared/data/cleaned_images10M/part2/sentinel_image_39.50451647812399_-119.77406874371808.tif",
"/shared/data/cleaned_images10M/part6/sentinel_image_8.883733122181614_9.692567308041912.tif",
"/shared/data/cleaned_images10M/part1/sentinel_image_37.200987879969695_-2.86372440726711.tif"
]

# Create a subplot to display multiple images
num_images = len(tif_files)
num_cols = 1
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and iterate through the images to display
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        tif_image = Image.open(tif_files[i])
        ax.imshow(tif_image)
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty subplots if there are fewer images than subplots

plt.tight_layout()  # Adjust the layout for better visualization
plt.show()


# In[9]:



import matplotlib.pyplot as plt
from PIL import Image

# Replace the file paths with the actual paths of your TIF images
tif_files = [
    "/shared/data/cleaned_images10M/part3/sentinel_image_29.75892242465566_-95.28527544639265.tif",
"/shared/data/cleaned_images10M/part1/sentinel_image_37.200987879969695_-2.86372440726711.tif"
]

# Create a subplot to display multiple images
num_images = len(tif_files)
num_cols = 1
num_rows = (num_images + num_cols - 1) // num_cols

# Set up the figure and iterate through the images to display
fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))

for i, ax in enumerate(axes.flatten()):
    if i < num_images:
        tif_image = Image.open(tif_files[i])
        ax.imshow(tif_image)
        ax.axis('off')
    else:
        ax.axis('off')  # Turn off axis for empty subplots if there are fewer images than subplots

plt.tight_layout()  # Adjust the layout for better visualization
plt.show()


# In[7]:


import pandas as pd
temp = pd.read_csv("image data testing 2.csv")
len(temp)


# In[33]:


df = pd.read_csv("PowerPlantDatabase.csv")
unique_df = df.drop_duplicates(subset='name', keep='first')
ml_df = unique_df.drop(columns=['name','geometry','Unnamed: 0'])
ml_df.reset_index(drop='True')


# In[35]:


import pandas as pd

# Assuming you have already loaded the DataFrame 'df'

# Shuffle the DataFrame to ensure randomization
df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Splitting 70% for training, 20% for testing, and 10% for validation
train_df = df.iloc[:int(0.7 * len(df))]
test_df = df.iloc[int(0.7 * len(df)):int(0.9 * len(df))]
validate_df = df.iloc[int(0.9 * len(df)):]

# Checking the sizes of the datasets
print(f"Train set size: {train_df.shape}")
print(f"Test set size: {test_df.shape}")
print(f"Validation set size: {validate_df.shape}")

train_df.to_csv("powerplant train.csv")
test_df.to_csv("powerplant test.csv")
validate_df.to_csv("powerplant validate.csv")


# In[2]:


pip install imagemagick 


# In[2]:


import re
def extract_coordinates_temp(file_name):


    pattern = r"[-+]?\d+\.\d+"

    matches = re.findall(pattern, filename)
    latitude = float(matches[0])
    longitude = float(matches[1])
    lat_lon_list = [latitude, longitude]

    return lat_lon_list


# In[3]:


import pandas as pd
from shapely import Point
import matplotlib as plt 

airport_df = pd.read_csv("AirportDatabase.csv")

powerplant_df = pd.read_csv("PowerPlantDatabase.csv")

airport_points = []
powerplant_points = []

for filename in airport_df['filename']:
    point = Point(extract_coordinates_temp(filename))
    airport_points.append(point)

for filename in powerplant_df['filename']:
    point = Point(extract_coordinates_temp(filename))
    powerplant_points.append(point)


# In[4]:


import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Step 1: Create GeoDataFrames from the existing lists of Point objects
airport_points_gdf = gpd.GeoDataFrame(geometry=airport_points)
powerplant_points_gdf = gpd.GeoDataFrame(geometry=powerplant_points)

# Step 2: Create a figure and axes for the plot
fig, ax = plt.subplots(figsize=(12, 8))

# Step 3: Plot the world map using a suitable basemap
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.plot(ax=ax, color='lightgray')

powerplant_points_gdf['swapped_geometry'] = powerplant_points_gdf.geometry.apply(lambda geom: Point(geom.y, geom.x))
powerplant_points_gdf_swapped = powerplant_points_gdf.set_geometry('swapped_geometry')
powerplant_points_gdf_swapped.plot(ax=ax, color='blue', markersize=5, label='Power Plants')

# Step 4: Swap the coordinates during plotting for both airport_points and powerplant_points
airport_points_gdf['swapped_geometry'] = airport_points_gdf.geometry.apply(lambda geom: Point(geom.y, geom.x))
airport_points_gdf_swapped = airport_points_gdf.set_geometry('swapped_geometry')
airport_points_gdf_swapped.plot(ax=ax, color='red', markersize=5, label='Airports')



# Step 5: Customize the plot appearance (if needed)
ax.set_title('Matched Airport Points and Power Plant Points on World Map')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend()

# Step 6: Show the plot
plt.show()


# In[14]:


powerplant_df = pd.read_csv("PowerPlantDatabase.csv")
airport_df = pd.read_csv("AirportDatabase.csv")
crop_df = pd.read_csv("CropHarvestDatabase.csv")
road1_df = pd.read_csv("Road1Database.csv")
mine_df = pd.read_csv("MineDatabase.csv")
offshore_df = pd.read_csv("OffshoreInstallationDatabase.csv")
glacier_df = pd.read_csv("GlacierDatabase.csv")
glacier2_df = pd.read_csv("Glacier2Database.csv")
port_df = pd.read_csv("PortDatabase.csv")
turbine_df = pd.read_csv("Wind TurbineDatabase.csv")
df = pd.concat([powerplant_df, airport_df,crop_df,road_df,road1_df,mine_df,offshore_df,glacier_df,glacier2_df,port_df,turbine_df], ignore_index=True)
ml_df = df.drop(columns=['name','Airport Name', 'Label', 'Country', 'mine_fac','Name', 'Glacier Name' , 'Main Port Name','geometry','Unnamed: 0'])
ml_df.tail(10)
ml_df.to_csv("Training Data.csv")

