import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from collections import defaultdict
import folium

def import_cbre(metro = None):
    cbre = pd.read_csv('../data/CBRE_Markets.csv',dtype={'zip_code':'object'})
    cbre.zip_code = cbre.zip_code.map(lambda z: '0'*(5-len(z))+z) # convert to standard 5-digit zip code
    cbre.set_index('zip_code',inplace=True)
    if metro:
        cbre = cbre[cbre['mktname'] == metro].copy()
    return(cbre)

def import_zillow(metro = None):
    zillow = pd.read_csv('../data/Zip_Zhvi_AllHomes.csv',dtype={'RegionName':'object'})
    zillow.columns = zillow.columns.str.replace('RegionName', 'zip_code')
    zillow.set_index('zip_code',inplace=True)
    if metro:
        zillow = zillow[zillow['Metro'] == metro].copy()
    zillow = zillow.iloc[:,np.r_[0:8,176:zillow.shape[1]]] # select the info part and 2010 - latest data
    return(zillow)

def plot_missing(data):
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    sns.heatmap(data.isnull(), cbar=False,ax=axs[0])
    axs[1].hist(data.isna().mean(axis=1),bins=20)

def split_zillow(zillow):
    return(zillow.iloc[:,:8].copy(),zillow.iloc[:,8:].copy())

def get_change(ts_data):
    ts = ts_data.T
    ts_pre = ts.shift()
    return(((ts-ts_pre)/ts_pre).iloc[1:,:].T)

def plot_inertia(km, X, n_cluster_range):
    inertias = []
    for i in n_cluster_range:
        km.set_params(n_clusters=i)
        km.fit(X)
        inertias.append(km.inertia_)
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    fig.suptitle('Elbow Method')
    axs[0].plot(n_cluster_range, inertias, marker='o')
    axs[0].set(xlabel='Number of clusters',ylabel='Inertia')
    axs[1].plot(n_cluster_range[1:], 
                [inertias[i+1]-inertias[i] for i in range(len(inertias)-1)], marker='o')
    axs[1].set(xlabel='Number of clusters',ylabel='Change of Inertia')
    
def fit_kmeans(ncluster,ts_data,km):
    km.set_params(n_clusters=ncluster)
    km.fit(ts_data)

    km_predict = pd.DataFrame({'zip_code':ts_data.index,'cluster':km.predict(ts_data)})

    return(km_predict)

def plot_clusters(km_predict,ts_data):
    nrow = -(-km_predict.cluster.nunique()//2)
    fig, axs = plt.subplots(ncols=2, nrows=nrow,figsize=(12, 4*nrow))
    for g in km_predict.groupby('cluster'):
        g_ts = ts_data[ts_data.index.isin(g[1].zip_code)].T
        g_ts.transform(np.log10).plot(ax=axs.flat[g[0]])
        axs.flat[g[0]].set_title('Cluster '+str(g[0]))
        axs.flat[g[0]].legend().set_visible(False)
        
def make_map(geojson,coor,ncluster,km_predict,zoom=8):
    
    geo_map = folium.Map(location=coor,zoom_start=zoom)

    folium.Choropleth(
        geo_data=geojson,
        data=km_predict,
        bins = np.arange(ncluster+1),
        name='choropleth',
        columns=km_predict.columns,
        key_on='feature.properties.ZCTA5CE10', # Use 'feature', not 'features'. this seems to be a bug
        fill_color='RdBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_opacity = 0,
        legend_name='Submarket'
    ).add_to(geo_map)

    folium.LayerControl().add_to(geo_map)
    return(geo_map)