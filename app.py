from flask import Flask, render_template, url_for, request, session, redirect
import numpy as np
import pandas as pd
import spotipy
from sklearn.datasets import load_iris
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
import base64
import random
from src.playlist_analysis import playlist_X, new_playlist
from spotipy.oauth2 import SpotifyClientCredentials
import matplotlib.cm as cm
from IPython.display import HTML, display
spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/results', methods=['POST'])
def results():
    
    filename = './models/knn_playlists.sav'
    knn = pickle.load(open(filename, 'rb'))
    sfilename = './models/scaler.sav'
    scaler = pickle.load(open(sfilename, 'rb'))
    playlist_master = pd.read_csv('./csv/playlist_master.csv', index_col=0)
    playlist_analysis = playlist_X(playlist_master)

    prediction = new_playlist(str(request.form['url']), playlist_analysis, playlist_master, knn, scaler)
          
    return render_template('results.html', prediction=[prediction.to_html(classes='data', header="true", index = False)])

