# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from flask import Flask, current_app
from apps.apriori import blueprint
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
from apyori import apriori
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

#Se establece la ruta /apriori
@blueprint.route('/apriori')
@login_required
def apriori():
    current_app.config["UPLOAD_FOLDER"] = "static/"
    return render_template('home/apriori.html', segment='apriori')


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
            segment = 'apriori'

        return segment

    except:
        return None

@blueprint.route('/apriori_resultados', methods = ['GET', 'POST'])
@login_required

def save_file():
    current_app.config["UPLOAD_FOLDER"] = "static/"
    
    if request.method == 'POST':
        f = request.files['file']

        filename = secure_filename(f.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))

        f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
        filepath=os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
        file = open("C:/Users/melan/Documents/flask-black-dashboard/apps/apriori/static/" + filename,"r")
        content = file.read()

        DatosTransacciones = pd.read_csv(filepath, header=None)
        #Se incluyen todas las transacciones en una sola lista
        Transacciones = DatosTransacciones.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida'
        Lista = pd.DataFrame(Transacciones)
        Lista['Frecuencia'] = 1
        #Se agrupa los elementos
        Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
        Lista = Lista.rename(columns={0 : 'Item'})
        plt.figure(figsize=(30,20), dpi=100)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
        plt.show()
        plt.savefig('C:/Users/melan/Documents/flask-black-dashboard/apps/apriori/static/my_plot.png')

    return render_template('home/content.html', filename =filename, content=content, get_plot = True, plot_url = 'C:/Users/melan/Documents/flask-black-dashboard/apps/apriori/static/my_plot.png') 
