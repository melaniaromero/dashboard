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
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

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

@blueprint.route('/apriori', methods = ['GET', 'POST'])
@login_required

def save_file():
    current_app.config["UPLOAD_FOLDER"] = "static/"
    
    if request.method == 'POST':
        f = request.files['file']

        filename = secure_filename(f.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))

        f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
        filepath=os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
        
        file = open("C:/Users/valer/Documents/GitHub/dashboard/apps/apriori/static/" + filename,"r")

        file = open(filepath, "r")
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
        fig= Figure ()
        axis = fig.add_subplot()
        fig.set_size_inches(16, 23)
        fig.set_dpi(700)
        axis.set_title("Distribución de la frecuencia de los elementos")
        axis.set_xlabel("Frecuencia")
        axis.set_ylabel("Item")
        axis.barh(Lista['Item'], width=Lista['Frecuencia'], color='violet')
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        FigureCanvas(fig).print_png(pngImage)
        # Encode PNG image to base64 string
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    return render_template('home/apriori.html', filename =filename, content=content, image=pngImageB64String)

@blueprint.route('/apriori_resultados', methods = ['GET', 'POST'])
@login_required

def save():
    from apyori import apriori
    current_app.config["UPLOAD_FOLDER"] = "static/"
    
    if request.method == 'POST':
        f = request.files['file']
        s = request.form['support']
        c =request.form['confidence']
        l =request.form['lift']

        filename = secure_filename(f.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))

        f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
        filepath=os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
        file = open(filepath, "r")
        content = file.read()

        DatosTransacciones = pd.read_csv(filepath, header=None)
        #Se incluyen todas las transacciones en una sola lista
        TransaccionesLista = DatosTransacciones.stack().groupby(level=0).apply(list).tolist()
        resultados = apriori(TransaccionesLista, min_support=float(s),  min_confidence=float(c),  
                             min_lift=int(l))
       
        ResultadosC1 = list(resultados) #lista de los resultados
        total_item = len(ResultadosC1)
        
        def inspect(ResultadosC1):
            baseItem   = [tuple(result[2][0][0])[0:] for result in ResultadosC1]
            addItem     = [tuple(result[2][0][1])[0] for result in ResultadosC1]
            supports    = [result[1] for result in ResultadosC1]
            confidences = [result[2][0][2] for result in ResultadosC1]
            lifts       = [result[2][0][3] for result in ResultadosC1]
            
            return list(zip(baseItem,addItem, supports, confidences, lifts))
        
        resultsinDataFrame = pd.DataFrame(inspect(ResultadosC1),columns=['items_base','items_add','Soporte',
                                                            'Confianza','Elevación'])

    return render_template('home/content.html',s=s,c=c,l=l,ResultadosC1=ResultadosC1,total_item=total_item,
                           column_names=resultsinDataFrame.columns.values, row_data=list(resultsinDataFrame.values.tolist()),
                           zip=zip)
