# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from flask import Flask, current_app
from apps.logistica import blueprint
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
import seaborn as sns
matplotlib.use('agg')
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay
#Se establece la ruta /metricas
@blueprint.route('/logistica')
@login_required
def logistica():
    current_app.config["UPLOAD_FOLDER"] = "static/"
    return render_template('home/logistica.html', segment='logistica')


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
            segment = 'logistica'

        return segment

    except:
        return None

@blueprint.route('/logistica', methods = ['GET', 'POST'])
@login_required

def save_file():
    current_app.config["UPLOAD_FOLDER"] = "static/"
    
    if request.method == 'POST':
        f = request.files['file']

        P = request.form['Pregnancies']
        G = request.form['Glucose']
        B= request.form['BloodPressure']
        S=request.form['SkinThickness']
        I=request.form['Insulin']
        BMI=request.form['BMI']
        D=request.form['DiabetesPedigreeFunction']
        E=request.form['Edad']



        filename = secure_filename(f.filename)

        basedir = os.path.abspath(os.path.dirname(__file__))

        f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
        filepath=os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
        file = open("C:/Users/valer/Documents/GitHub/dashboard/apps/logistica/static/" + filename,"r")
        Diabetes = pd.read_csv(filepath)

        out=Diabetes.groupby('Outcome').size()
        outD=pd.DataFrame(out)

        MatrizInf = np.triu(Diabetes.corr())
        
        sns.heatmap(Diabetes.corr(), cmap='RdBu_r', annot=True, mask=MatrizInf)
        img = io.BytesIO()
        plt.gcf().set_size_inches(16, 18)
        plt.savefig(img, format='png',  dpi=300)
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

       
        #Variables predictoras
        X = np.array(Diabetes[['Pregnancies', 
                       'Glucose', 
                       'BloodPressure', 
                       'SkinThickness', 
                       'Insulin', 
                       'BMI',
                       'DiabetesPedigreeFunction',
                       'Age']])
        
        #Variable clase
        Y = np.array(Diabetes[['Outcome']])
     



        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 0,
                                                                                shuffle = True)
        
        ClasificacionRL = linear_model.LogisticRegression()
        ClasificacionRL.fit(X_train, Y_train)
        Probabilidad = ClasificacionRL.predict_proba(X_validation)

        proba=pd.DataFrame(Probabilidad)
        
        Y_ClasificacionRL = ClasificacionRL.predict(X_validation)
        accuracy=accuracy_score(Y_validation, Y_ClasificacionRL)

        ModeloClasificacion = ClasificacionRL.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                   ModeloClasificacion, 
                                   rownames=['Reales'], 
                                   colnames=['Clasificación']) 
        
        exactitud=accuracy_score(Y_validation, Y_ClasificacionRL)
        ab=classification_report(Y_validation, Y_ClasificacionRL, output_dict=True)
        report = pd.DataFrame(ab).transpose()

        CurvaROC = RocCurveDisplay.from_estimator(ClasificacionRL, X_validation, Y_validation, name="Diabetes")


        img1 = io.BytesIO()
        plt.gcf().set_size_inches(16, 18)
        plt.savefig(img1, format='png',  dpi=300)
        plt.close()
        img.seek(0)
        plot_url1 = base64.b64encode(img1.getvalue()).decode('utf8')


        #Paciente
        PacienteID = pd.DataFrame({'Pregnancies': [int(P)],
                           'Glucose': [int(G)],
                           'BloodPressure': [int(B)],
                           'SkinThickness': [int(S)],
                           'Insulin': [int(I)],
                           'BMI': [float(BMI)],
                           'DiabetesPedigreeFunction': [float(D)],
                           'Age': [int(E)]})
        resultadosF=ClasificacionRL.predict(PacienteID)


        






        return render_template('home/logistica.html', filename =filename, plot_url=plot_url, outD=outD.to_html(),
                               proba=proba.to_html(), Y_ClasificacionRL= Y_ClasificacionRL,
                               accuracy=accuracy, Matriz_Clasificacion= Matriz_Clasificacion.to_html(),
                               resultadosF=resultadosF,exactitud=exactitud, report=report.to_html(), plot_url1=plot_url1)



       
        
        
        
        
        




