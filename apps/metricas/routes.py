# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from flask import Flask, current_app
from apps.metricas import blueprint
from flask_login import login_required
from jinja2 import TemplateNotFound
import os
from itertools import chain, combinations
from collections import defaultdict
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import datetime
import pandas as pd 
import numpy as np                  # Para crear vectores y matrices n dimensionales
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler  

#Se establece la ruta /metricas
@blueprint.route('/metricas')
@login_required
def metricas():
    current_app.config["UPLOAD_FOLDER"] = "static/"
    return render_template('home/metricas.html', segment='metricas')


@blueprint.route('/<template>')
@login_required
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'metricas'

        return segment

    except:
        return None

@blueprint.route('/metricas', methods = ['GET', 'POST'])
@login_required

def save_file():
    current_app.config["UPLOAD_FOLDER"] = "static/"
    
    if request.method == 'POST':
        f = request.files['file']
        fil =request.form['fila']
        filas = int(fil)

        filename = secure_filename(f.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))

        f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
        filepath=os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
        file = open("C:/Users/melan/Documents/flask-black-dashboard/apps/metricas/static/" + filename,"r")
        Hipoteca = pd.read_csv(filepath)

        select1= request.form.get('estandarizar')
        select2= request.form.get('distancia')

        if select1== "Normalizada" and select2== "Euclidiana":
            normalizar=MinMaxScaler()
            MNormalizada=normalizar.fit_transform(Hipoteca)
            #EUCLIDIANA
            #DstNEuclidiana = cdist(MNormalizada, MNormalizada, metric='euclidean')
            #MNEuclidiana = pd.DataFrame(DstNEuclidiana)
            DstNEuclidiana = cdist(MNormalizada[0:filas], MNormalizada[0:filas], metric='euclidean')
            MNEuclidiana = pd.DataFrame(DstNEuclidiana)
            return render_template('home/normaEuclidiana.html', filename =filename, MNEuclidiana=MNEuclidiana.to_html())
        




