import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from collections import defaultdict
import folium
from sklearn.preprocessing import StandardScaler
import json

def import_centroid():
    """ Import centroid data from ../data/ZIP_centroid.csv """
    zip_centroid = pd.read_csv('../data/ZIP_centroid.csv',dtype={'zip':'object'})
    zip_centroid = zip_centroid[['zip','longitude','latitude']].copy()
    zip_centroid.columns = zip_centroid.columns.str.replace('zip', 'zip_code')
    zip_centroid.zip_code = zip_centroid.zip_code.map(lambda z: z.zfill(5))
    return(zip_centroid)

def import_pop():
    """ Import population data from ../data/ZCTA_Population.csv """
    zip_pop = pd.read_csv('../data/ZCTA_Population.csv',dtype={'ZCTA':'object'})
    zip_pop = zip_pop[['ZCTA','Population','Household_Count']].copy()
    zip_pop.columns = zip_pop.columns.str.replace('ZCTA', 'zip_code')
    zip_pop.zip_code = zip_pop.zip_code.map(lambda z: z.zfill(5))
    return(zip_pop)

def combine_geo(in_json):
    """ 
    Combine geo data from multiple states togethter.
    
    Input: a list of file names of geo JSON file for each state.
    """
    list_geo = []  

    for i in range(len(in_json)):
        with open(in_json[i],'r') as jsonFile:
            state = json.load(jsonFile)
        for j in range(len(state['features'])):
            list_geo.append(state['features'][j])

    metro_geo = dict.fromkeys(['type', 'features'])
    metro_geo['type'] = 'FeatureCollection'
    metro_geo['features'] = list_geo
    
    return(metro_geo)

def get_centroid(data,zip_centroid):
    """
    Get the centroids for each submarket
    
    Input:
    data: a DataFrame that includes two columns, zip_code and submktname.
    zip_centroid: output of import_centroid that has centroid coordinates.
    
    Output:
    Centroid coordinates for each submarket.
    
    """
    centroid = pd.merge(data,zip_centroid,on='zip_code'
                        ).groupby('submktname').agg({'latitude':'median',
                                                     'longitude':'median'})
    return(centroid)

def get_stat(data,zip_pop):
    """
    Get the statistics for each submarket
    
    Input:
    data: a DataFrame that includes two columns, zip_code and submktname.
    zip_pop: output of import_pop that has population and household counts data.
    
    Output:
    Statistics of population and household counts for each submarket.
    
    """
    
    stat = pd.merge(data,zip_pop,on='zip_code'
                        ).groupby('submktname').agg({'Population':'median',
                                                     'Household_Count':'median'})
    stat = pd.concat([data.submktname.value_counts(),stat],axis=1,sort=True)
    return(stat)
    
def import_cbre(metro = None):
    """
    Import CBRE data. 
    
    Input:
    metro: name of the metro area. Default None, the entire CBRE data will be returned.
    """
    
    cbre = pd.read_csv('../data/CBRE_Markets.csv',dtype={'zip_code':'object'})
    cbre.zip_code = cbre.zip_code.map(lambda z: z.zfill(5)) # convert to standard 5-digit zip code
    cbre.set_index('zip_code',inplace=True)
    if metro:
        cbre = cbre[cbre['mktname'] == metro].copy()
    return(cbre)

def import_zillow(metro = None, zip_code = None):
    """
    Import Zillow data. 
    
    Input:
    metro: name of the metro area.
    zip_code: Zip Codes of selected area.
    """
    
    zillow = pd.read_csv('../data/Zip_Zhvi_AllHomes.csv',dtype={'RegionName':'object'})
    zillow.columns = zillow.columns.str.replace('RegionName', 'zip_code')
    zillow.set_index('zip_code',inplace=True)
    if metro:
        zillow = zillow[zillow['Metro'] == metro].copy()
    if zip_code is not None:
        zillow = zillow[zillow.index.isin(zip_code)].copy()
    zillow = zillow.iloc[:,np.r_[0:8,176:zillow.shape[1]]] # select the info part and 2010 - latest data
    return(zillow)

def plot_missing(data):
    """
    Visualize missingness using seaborn.
    """
    
    fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
    sns.heatmap(data.isnull(), cbar=False,ax=axs[0])
    axs[1].hist(data.isna().mean(axis=1),bins=20)

def split_zillow(zillow):
    """
    Split the information section and time series section in Zillow data.
    """
    return(zillow.iloc[:,:8].copy(),zillow.iloc[:,8:].copy())

def get_change(ts_data):
    """
    Calculate percentage changes of time series data.
    """
    
    ts = ts_data.T
    ts_pre = ts.shift()
    return(((ts-ts_pre)/ts_pre).iloc[1:,:].T)

def get_syn(change,zip_centroid,w):
    """ 
    Combine time series info and centroid info for each Zip Code. 
    
    Input:
    change: percentage change of time series data
    zip_centroid: output of import_centroid
    w: weight of centroid information
    """
    
    zillow_zip_centroid = (pd.merge(change,zip_centroid,on='zip_code').set_index('zip_code')).iloc[:,-2:]
    scaler = StandardScaler()
    zillow_zip_centroid = pd.DataFrame(scaler.fit_transform(zillow_zip_centroid),columns=zillow_zip_centroid.columns,
                                   index=zillow_zip_centroid.index)
    weight = change.std().mean()*w
    syn = pd.merge(change,(zillow_zip_centroid*weight).reset_index(),on='zip_code').set_index('zip_code')
    return(syn)

def plot_inertia(km, X, n_cluster_range):
    """
    Plot the inertia of k-means clustering for a range of cluster numbers.
    
    Input:
    km: k-means clustering
    X: data
    n_cluster_range: number range
    """
    
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
    """
    Fit k-means clustering and return the result in a DataFrame
    
    Input:
    ncluster: number of clusters
    ts_data: time series data
    km: k-means clustering
    
    Output:
    A DataFrame with zip_code as row index and submktname (cluster) as column.
    """
    
    km.set_params(n_clusters=ncluster)
    km.fit(ts_data)

    km_predict = pd.DataFrame({'zip_code':ts_data.index,'submktname':km.predict(ts_data)})

    return(km_predict)

def plot_clusters(km_predict,ts_data,log=True):
    """
    Plot time series curves of each cluster.
    
    Input:
    km_predict: output of fit_kmeans
    ts_data: time series data
    log: plot in logarithmic scale. Default True.
    """
    
    nrow = -(-km_predict.submktname.nunique()//2)
    fig, axs = plt.subplots(ncols=2, nrows=nrow,figsize=(12, 4*nrow))
    for i,g in enumerate(km_predict.groupby('submktname')):
        g_ts = ts_data[ts_data.index.isin(g[1].zip_code)].T
        if log==True:
            g_ts.transform(np.log10).plot(ax=axs.flat[i])
        else:
            g_ts.plot(ax=axs.flat[i])
        axs.flat[i].set_title('Cluster '+str(g[0]))
        axs.flat[i].legend().set_visible(False)

        
def make_map(geojson,coor,ncluster,cluster,zoom=8,centroid=None):
    """
    Make a map using Folium.
    
    Input:
    geojson: geo json data
    coor: center coordinates of the map
    ncluster: number of clusters
    cluster: output of fit_kmeans
    zoom: zoom level
    centroid: centroid coordinates of submarkets
    """
    
    geo_map = folium.Map(location=coor,zoom_start=zoom)

    choropleth = folium.Choropleth(
        geo_data=geojson,
        data=cluster,
        bins = np.arange(ncluster+1),
        name='choropleth',
        columns=cluster.columns,
        key_on='feature.properties.ZCTA5CE10', # Use 'feature', not 'features'. this seems to be a bug
        fill_color='RdBu',
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_opacity = 0.5,
        legend_name='Submarket',
        highlight = True
    ).add_to(geo_map)

    # add labels
    # See https://towardsdatascience.com/choropleth-maps-with-folium-1a5b8bcdd392
    style_function = "font-size: 15px; font-weight: bold"
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['ZCTA5CE10'], style=style_function, labels=False))

    folium.LayerControl().add_to(geo_map)
    
    if centroid is not None:
        for i in range(centroid.shape[0]):
            folium.Marker(centroid.iloc[i,:].to_list(),popup=centroid.index[i]).add_to(geo_map)
                
    return(geo_map)