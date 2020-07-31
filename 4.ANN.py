#%%
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

df = pd.read_csv('data/NYCTaxiFare.csv')
print(df.head)
print(df['fare_amount'].describe())

#Calculating haversine_distance

def haversine_distance(df,  lat1, long1, lat2, long2):
    """
        Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371 #radius of earth in km
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    delta_phi = np.radians(df[lat2] - df[lat1])
    delta_lambda = np.radians(df[long2] - df[long1])
    a = np.sin(delta_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (r * c)  # in kilometers
    return d

df['dist_km'] = haversine_distance(df, 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude')
print(df.head())

#Add a datetime column and derive useful statistics
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
print(df.info())

df['EDTdate'] = df['pickup_datetime']-pd.Timedelta(hours=4)
df['Hour'] = df['EDTdate'].dt.hour
df['AMorPM'] = np.where(df['Hour']<12,'am','pm')
df['Weekday'] = df['EDTdate'].dt.strftime("%a")
print(df.head())
print(df['EDTdate'].min())
print(df['EDTdate'].max())
print(df.columns)

# Separate categorical from continuous columns
cat_cols = ['Hour', 'AMorPM', 'Weekday']
cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'dist_km']
y_col = ['fare_amount'] #label

# Convert our three categorical columns to category dtypes.
for cat in cat_cols:
    df[cat] = df[cat].astype('category')
print(df.dtypes)
print(df.head())
print(df['Hour'].head())
print(df['AMorPM'].head())
print(df['AMorPM'].cat.categories)
print(df['AMorPM'].head().cat.codes)
print(df['Weekday'].cat.categories)
print(df['Weekday'].head().cat.codes)

# combine the three categorical columns into one input array using numpy.stack

hr = df['Hour'].cat.codes.values
ampm = df['AMorPM'].cat.codes.values
wkdy = df['Weekday'].cat.codes.values

cats = np.stack([hr, ampm, wkdy], 1)

print(cats[:5])
# OR->
# cats = np.stack([df[col].cat.codes.values for col in cat_cols], 1)























