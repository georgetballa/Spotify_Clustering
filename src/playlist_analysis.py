import pandas as pd
import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import os.path
from os import path
import matplotlib.pyplot as plt

import json

import itertools
import scipy.stats as scs
from scipy.spatial.distance import pdist, squareform

from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import pickle

import matplotlib.cm as cm
from IPython.display import HTML, display
spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

import pickle

from mpl_toolkits.mplot3d import Axes3D
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

### Comment this out to work with other playlist csv
playlist_master = pd.read_csv('./csv/playlist_master.csv', index_col=0)

### Pipeline for JSON to pandas DF
def json_concat_pipeline(min_followers, num_tracks, first, last):
    
    '''Return Pandas DF of playlists with min number of followers, with first, last to set range 
    
    min_num_followers: minimum number of followers a playlist must have to make the list
    first: first document to pull from (starts at 0)
    last: inclusive last document (a last of n value will have a doc with name range mpd.slice.n*1000 - n*1000+999.json)
    
    '''
    df = pd.DataFrame()
    start = 1000*first
    for i in range(first,last+1):
        data = json.load(open(f'./data/mpd.slice.{start}-{start+999}.json'))
        df_pl = pd.DataFrame(data["playlists"])
        filt = (df_pl['num_followers'] >= min_followers) & (df_pl['num_tracks'] >= num_tracks)
        df_filt = df_pl[filt]
        df_filt.drop(['pid','description'], axis=1, inplace=True)
        start += 1000
        df = df.append(df_filt)
        
#     df.to_csv(f'./csv/playlist_{min_followers}f_{num_tracks}t_{first}_{last}.csv')
                
    return df

def get_all_tracks_csv(first, last):
    '''Will request all tracks from playlists in json database from first document to last'''
    start = 1000*first
    for i in range(first,last):
        data = json.load(open(f'./data/mpd.slice.{start}-{start+999}.json'))
        df_pl = pd.DataFrame(data["playlists"])
        df_pl.drop(columns='description', inplace=True)
        df_pl.dropna(inplace=True)
        songs = pd.DataFrame(columns=['track_uri'])
                      
        
        for j in range(df_pl.shape[0]):
            tracks = pd.DataFrame(df_pl.iloc[j]['tracks'])
            tracks = tracks[['artist_name','track_name','duration_ms', 'album_name', 'track_uri']]
            tracks.dropna(inplace=True)
                                    
            splits = len(tracks.track_uri)
            if splits>200:
                num_splits = (splits//150)
            else:
                num_splits = 2
                
            start1 = 0
            end1 = splits//num_splits
            for k in range(2,num_splits+1):
                tracklist = list(tracks.track_uri)[start1:end1]                             
                       
                start1 = start1+end1+1
                end1 = end1*k
                if len(tracklist) > 0:
                    track_req = spotify.audio_features(tracklist)
                    
                    tracks_df = pd.DataFrame(track_req)      
                    tracks_df = tracks_df.rename(columns={"uri": "track_uri"})
                    tracks_df.drop(['id', 'track_href', 'analysis_url', 'duration_ms'], axis=1, inplace=True)
                    songs = songs.append(tracks_df)
                    
                
        songs = songs.drop_duplicates(subset='track_uri')
        songs.drop(columns= ['type'], inplace=True)
        songs.to_csv(f'./csv/songs{start}-{start+999}.csv')
        start += 1000
    return songs

def batch_track_to_csv(first, last, interval):
    '''Append multiple tracks to one CSV'''
    df = pd.DataFrame()
    start = interval*first
    for i in range(first,last+1):
        if path.exists(f'./csv/songs{start}-{start+999}.csv'):
            data = pd.read_csv(f'./csv/songs{start}-{start+999}.csv')
            start += interval
            df = df.append(data)
            
        else:
            start += interval
    df = df.drop_duplicates(subset='track_uri')
    df.to_csv(f'./csv/songs{first*interval}-{(last*interval)-1}.csv')

def playlist_songs(df, songs):
    '''
    Creates a CSV for all playlists containing song data for songs in the playlist  

    Used to create playlist_master CSV
    
    '''
    
    
    
    for i in range(df.shape[0]):
        playlist_tracks = pd.DataFrame(df.iloc[i]['tracks'])
        playlist_tracks.rename(columns={'pos':'playlist_name'}, inplace=True)
        playlist_tracks['playlist_name'] = df.iloc[i]['name']
        playlist_tracks['num_tracks'] = df.iloc[i]['num_tracks']
        playlist_tracks['num_albums'] = df.iloc[i]['num_albums']
        playlist_tracks['num_followers'] = df.iloc[i]['num_followers']
        playlist_tracks = playlist_tracks.merge(songs, how='left', on = 'track_uri')
        
        playlist_tracks.to_csv(f'./csv/playlist_master.csv', mode='a', header=False)
        if i%100 == 0:
            print(i)
    return 

def merge_two_csv(first_path, second_path, destination_path):
    '''Merge two CSV files'''
    df = pd.read_csv(first_path, index_col=0)
    df2 = pd.read_csv(second_path, index_col=0)
    df = df.append(df2)
    df = df.drop_duplicates(subset='track_uri')
    df.to_csv(destination_path)



def playlist_X(df):
    '''
    Format playlist with numerical song analysis data to playlist stats DF for X input matrix
    
    
    '''
    

    playlist_X = df[['playlist_name','danceability',
                     'key','loudness','mode','speechiness','acousticness','instrumentalness',
                     'liveness','valence','tempo', 'num_tracks']]
    playlist_X_stats = playlist_X.groupby(['playlist_name','num_tracks']).agg({'danceability': [np.std, np.mean],
                                                                'loudness': [np.std, np.mean],
                                                                'mode': [np.std, np.mean],
                                                                'speechiness': [np.std, np.mean],
                                                                'acousticness': [np.std, np.mean],
                                                                'instrumentalness': [np.std, np.mean],
                                                                'liveness': [np.std, np.mean],
                                                                'valence': [np.std, np.mean],
                                                                'tempo': [np.std, np.mean]}).reset_index(level=1)
    playlist_X_stats.dropna(inplace=True)
    playlist_X_stats.drop(columns='num_tracks', inplace=True)
    return playlist_X_stats

def cluster_testing(df_scaled_pca, maxk, model):
    '''Test range of values of K for kmeans clustering'''

    x = np.array(mstats)
    wcss = np.zeros(maxk)
    silhouette = np.zeros(maxk)

#     fig, axes = plt.subplots(5, 4, figsize=(16,9))

    # flatten
#     axes = [ax for axrow in axes for ax in axrow]

    for k in range(1,maxk):
        
        y = model.fit_predict(x)
    #     ax.axis('off')
    #     ax.scatter(x[:,0], x[:,1], c=y, linewidths=0, s=10)
    #     ax.set_ylim(ymin=-9, ymax=8)


        for c in range(0, k):
            for i1, i2 in itertools.combinations([ i for i in range(len(y)) if y[i] == c ], 2):
                wcss[k] += sum(x[i1] - x[i2])**2
        wcss[k] /= 2

        if k > 1:
            silhouette[k] = silhouette_score(x,y)
    
    fig, ax = plt.subplots()
    ax.plot(range(2,maxk), wcss[2:maxk], 'o-')
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("within-cluster sum of squares")
    plt.title('Inertia Graph')
    
    fig2, ax2 = plt.subplots()
    ax2.plot(range(2,maxk), silhouette[2:maxk], 'o-')
    ax2.set_xlabel("number of clusters")
    ax2.set_ylabel("silhouette score")
    #ax.set_ylim(ymin=0.0, ymax=1.0)

    return (maxk, wcss, silhouette)

def mixture_testing(df_scaled_pca, mink, maxk):
    '''Testing cluster numbers for Gaussian mixture models'''

    x = np.array(df_scaled_pca)
    aic = np.zeros(maxk)
    score = np.zeros(maxk)
    converged = np.zeros(maxk)
    silhouette = np.zeros(maxk)
    
#     fig, axes = plt.subplots(5, 4, figsize=(16,9))

    # flatten
#     axes = [ax for axrow in axes for ax in axrow]

    for k in range(mink,maxk):
        model = GaussianMixture(n_components=k, n_init=15, max_iter=300)
        y = model.fit_predict(x)
    #     ax.axis('off')
    #     ax.scatter(x[:,0], x[:,1], c=y, linewidths=0, s=10)
    #     ax.set_ylim(ymin=-9, ymax=8)
                
        if k > 1:
            aic[k] = model.aic(x)
            b = aic[aic!=0]
            converged[k] = model.converged_
#             silhouette[k] = silhouette_score(x,y)
            print(k)
            np.append(aic_tot, aic[k])
    fig, ax = plt.subplots()
    ax.plot(range(mink ,maxk), aic_tot[mink :maxk], 'o-')
    ax.set_xlabel("number of clusters")
    ax.set_ylabel("AIC Score (lower is better)")
    plt.title('Akaike Information Criterion')
    plt.tight_layout()
    plt.savefig(f'./img/gm_{mink}-{maxk}clusters_full.png', dpi=200)
    return aic, converged

def new_playlist(uri, playlist_analysis, playlist_master, knn, scaler):
    '''Evaluate new playlist and return nearest neighbors'''
    pl = spotify.playlist_items(uri, fields=None, market=None)
    plt = pd.DataFrame(pl['items'])
    lst = []
    songs = pd.DataFrame(columns=['track_uri'])
    for i in range(plt.shape[0]):
        lst.append(plt['track'][i]['uri'])
    
    splits = len(lst)
    if splits>200:
        num_splits = (splits//150)
    else:
        num_splits = 2

    start1 = 0
    end1 = splits//num_splits
    for k in range(2,num_splits+1):
        tracklist = lst[start1:end1]                             

        start1 = start1+end1+1
        end1 = end1*k
        if len(tracklist) > 0:
            track_req = spotify.audio_features(tracklist)

            tracks_df = pd.DataFrame(track_req)      
            tracks_df = tracks_df.rename(columns={"uri": "track_uri"})
            tracks_df.drop(['id', 'track_href', 'analysis_url', 'duration_ms'], axis=1, inplace=True)
            songs = songs.append(tracks_df)


    songs = songs.drop_duplicates(subset='track_uri')
    songs.drop(columns= ['type'], inplace=True)
    songs = songs[['danceability',
                     'key','loudness','mode','speechiness','acousticness','instrumentalness',
                     'liveness','valence','tempo']]
    songs = songs.agg({'danceability': [np.std, np.mean],
                        'loudness': [np.std, np.mean],
                        'mode': [np.std, np.mean],
                        'speechiness': [np.std, np.mean],
                        'acousticness': [np.std, np.mean],
                        'instrumentalness': [np.std, np.mean],
                        'liveness': [np.std, np.mean],
                        'valence': [np.std, np.mean],
                        'tempo': [np.std, np.mean]})
    songs = pd.DataFrame(songs.T.stack()).T
    songs = scaler.transform(np.array(songs).reshape(1,-1))
    indices = knn.kneighbors(songs, 5, return_distance=False)
            
    neighbors = pd.DataFrame()
    for i in indices[0]:
        neighbors = neighbors.append(playlist_master[playlist_master['playlist_name']==playlist_analysis.iloc[i].name])
        
    return neighbors[['artist_name','track_name']]

def cat_dict(centers):
    '''Generate dictionary of categories determined by KNN
    Make sure to use full set of features not just PCA or Scaler Error will occur
    
    centers: cluster centers from clustering model
    
    '''
    d = {}
    for i in range(len((centers))):
        center = scaler.transform(np.array(centers[i]).reshape(1,-1))
        indices = knn.kneighbors(center, 5, return_distance=False)
        neighbors = pd.DataFrame()
        for j in indices[0]:
            neighbors = neighbors.append(playlist_master[playlist_master['playlist_name']==playlist_analysis.iloc[j].name])

        d[i] = neighbors
    return d

def see_category_artists(d, cat_num):
    ''' View most common artists in category
    
    d: dictionary of cluster dataframes with category is key
    cat_num: Category number
    
    '''
    
    for i in d[cat_num].groupby('artist_name').count().sort_values(by='playlist_name', ascending=False).index[:50]:
        print(i)

def see_category(cat_num):
    '''view category DataFrame'''
    return d[cat_num][['playlist_name','artist_name','track_name', 'album_name']].head(500)

def run_gaussian_mixing(arr):
    gm = GaussianMixture(n_components=30, n_init=15, max_iter=300)
    y = gm.fit_predict(arr)
    err = gm.aic(arr)
    return y,err

def run_knn(arr):
    knn = NearestNeighbors(n_neighbors=5, algorithm = 'auto')
    knn.fit(arr)
