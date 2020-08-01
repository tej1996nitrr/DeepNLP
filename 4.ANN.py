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

# Convert numpy arrays to tensors
cats = torch.tensor(cats, dtype=torch.int64)
cats[:5]
conts = np.stack([df[col].values for col in cont_cols], 1)
conts = torch.tensor(conts, dtype=torch.float)
conts[:5]
conts.type()
y = torch.tensor(df[y_col].values, dtype=torch.float).reshape(-1,1) #reshape to make sure to keep column shape instead of flattened array
y[:5]
print(cats.shape)
print(conts.shape)
print(y.shape)

# Set an embedding size
# This will set embedding sizes for Hours, AMvsPM and Weekdays
cat_szs = [len(df[col].cat.categories) for col in cat_cols]
print(cat_szs)
emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
emb_szs

'''Tabular Model'''

# This is our source data
catz = cats[:4]
print(catz)

# This is passed in when the model is instantiated
print(emb_szs)
print([nn.Embedding(ni,nf) for ni,nf in emb_szs])

# This is assigned inside the __init__() method
selfembeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
print(selfembeds)
print(list(enumerate(selfembeds)))

# This happens inside the forward() method
embeddingz = []
for i,e in enumerate(selfembeds):
    embeddingz.append(e(catz[:,i]))
print(embeddingz)

# We concatenate the embedding sections (12,1,4) into one (17)
z = torch.cat(embeddingz, 1)
print(z)

# This was assigned under the __init__() method
selfembdrop = nn.Dropout(.4)
z = selfembdrop(z)
print(z)


class TabularModel(nn.Module):
    def init(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        """:arg
        emb_szs: list of tuples: each categorical variable size is paired with an embedding size
        n_cont: int: number of continuous variables
        out_sz: int: output sizees
        layers: list of ints: layer size
        p: float: dropout probability for each layer
        """
        super.__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs]) #embeddings
        self.emb_drop = nn.Dropout(p)  # dropuot
        self.bn_cont = nn.BatchNorm1d(n_cont)  # normalization
        layer_list = [] #storing the layers
        n_emb = sum([nf for ni, nf in emb_szs]) # sum of total embeddings
        n_in = n_emb + n_cont # number of inputs

        # create identical layers with sequence of operations, e.g.
        # layers = [100, 50, 25]
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))  # final layer

        # assign layers to atributes
        self.layers = nn.Sequential(*layerlist)

    def forward(self,x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)  # categorical embeddings

        x_cont = self.bn_cont(x_cont)  # continuous features
        x = torch.cat([x, x_cont], 1)  # concatenate categorial and continuous features
        x = self.layers(x)  # apply layers
        return x


layerlist = []  # storing the layers
p = 0.5
layers = [100, 50, 25]
n_in = 200
for i in layers:
    layerlist.append(nn.Linear(n_in, i))
    layerlist.append(nn.ReLU(inplace=True))
    layerlist.append(nn.BatchNorm1d(i))
    layerlist.append(nn.Dropout(p))
    n_in = i

print(layerlist)
nn.Sequential(*layerlist)

ni = 20
nf = 10
print(nn.Embedding(ni, nf))

test_emb =nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
print(test_emb)
for i,e in enumerate(test_emb):
    print(i,e)














