#%%
import torch
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import time

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

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_szs])  # embeddings
        self.emb_drop = nn.Dropout(p)  # dropuot
        self.bn_cont = nn.BatchNorm1d(n_cont)  # normalization

        layerlist = []  # storing the layers
        n_emb = sum((nf for ni, nf in emb_szs))  # sum of total embeddings
        n_in = n_emb + n_cont  # number of inputs

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

    def forward(self, x_cat, x_cont):
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


torch.manual_seed(33)
model = TabularModel(emb_szs, conts.shape[1], 1, [200,100], p=0.4)

model

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 60000 # two batches
test_size = int(batch_size * .2)

#data already shuffled
cat_train = cats[:batch_size-test_size]
cat_test = cats[batch_size-test_size:batch_size]
con_train = conts[:batch_size-test_size]
con_test = conts[batch_size-test_size:batch_size]
y_train = y[:batch_size-test_size]
y_test = y[batch_size-test_size:batch_size]

len(cat_train)

len(cat_test)


start_time = time.time()
epochs = 300
losses = []

for i in range(epochs):
    i+=1
    y_pred = model(cat_train, con_train)
    loss = torch.sqrt(criterion(y_pred, y_train)) # RMSE
    losses.append(loss)

    # a neat trick to save screen space:
    if i%25 == 1:
        print(f'epoch: {i:3}  loss: {loss.item():10.8f}')

    optimizer.zero_grad()   # reset gradients to 0
    loss.backward()
    optimizer.step()

print(f'epoch: {i:3}  loss: {loss.item():10.8f}') # print the last line
print(f'\nDuration: {time.time() - start_time:.0f} seconds') # print the time

# plt.plot(range(epochs), losses)
# plt.ylabel('RMSE Loss')
# plt.xlabel('epoch');

#validate model
with torch.no_grad():
    y_val = model(cat_test, con_test)
    loss = torch.sqrt(criterion(y_val, y_test))
print(f'RMSE: {loss:.8f}')



