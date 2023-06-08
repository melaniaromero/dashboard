from flask import Flask, render_template, request, current_app
from apps.arboles import blueprint
from flask_login import login_required
import os
from jinja2 import TemplateNotFound
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix, accuracy_score
from sklearn import model_selection
import seaborn as sns
from werkzeug.utils import secure_filename

# Ruta de inicio
@blueprint.route('/arboles')
@login_required
def index():
    return render_template('home/arboles.html')

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
            segment = 'index'
        return segment
    except:
        return None

@blueprint.route('/arboles', methods=['POST'])
@login_required
def arboles():
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        basedir = os.path.abspath(os.path.dirname(__file__))
        f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
        filepath = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename) 
        
        datos = pd.read_csv(filepath)
        variables_borrar_ar = request.form.getlist('variables_eliminar')
        for borrar in variables_borrar_ar:
            datos = datos.drop(columns=[borrar])
        
        var_clase = request.form['variable_clase']
        datos_sin_clase = datos.drop(columns=[var_clase])
        
        X = np.array(datos_sin_clase[datos_sin_clase.columns.values])
        Y = np.array(datos[[var_clase]])
        
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0, shuffle=True)
        
        clas_option1 = request.form['tipo_arbol']
        
        if clas_option1 == "Pronóstico":
            clas_option = request.form['modo_entrenamiento']
            
            if clas_option == "Modo 1 con random_state=0":
                PronosticoAD = DecisionTreeRegressor(random_state=0)
                PronosticoAD.fit(X_train, Y_train)
                Y_Pronostico = PronosticoAD.predict(X_test)
                
                score = r2_score(Y_test, Y_Pronostico) * 100
                criterion = PronosticoAD.criterion
                importancia_variables = PronosticoAD.feature_importances_
                mae = mean_absolute_error(Y_test, Y_Pronostico)
                mse = mean_squared_error(Y_test, Y_Pronostico)
                rmse = mean_squared_error(Y_test, Y_Pronostico, squared=False)
                
                ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]), 'Importancia': importancia_variables}).sort_values('Importancia', ascending=False)
                
                pred = {}
                for i in datos_sin_clase.columns.values:
                    pred[i] = request.form[i]
                pred = pd.DataFrame([pred])
                prediccion = PronosticoAD.predict(pred)
                
                return render_template('home/resultado_pronostico.html', score=score, criterion=criterion, importancia_variables=importancia_variables, mae=mae, mse=mse, rmse=rmse, importancia_df=ImportanciaMod2.to_html(), prediccion=prediccion)
                
            if clas_option == "Modo 2 con parámetros de número hojas, muestras y niveles del árbol":
                maxdepth = int(request.form['max_depth'])
                split = int(request.form['split'])
                leaf = int(request.form['leaf'])
                random = int(request.form['random_state'])
                
                PronosticoAD = DecisionTreeRegressor(max_depth=maxdepth, min_samples_split=split, min_samples_leaf=leaf, random_state=random)
                PronosticoAD.fit(X_train, Y_train)
                Y_Pronostico = PronosticoAD.predict(X_test)
                
                score = r2_score(Y_test, Y_Pronostico) * 100
                criterion = PronosticoAD.criterion
                importancia_variables = PronosticoAD.feature_importances_
                mae = mean_absolute_error(Y_test, Y_Pronostico)
                mse = mean_squared_error(Y_test, Y_Pronostico)
                rmse = mean_squared_error(Y_test, Y_Pronostico, squared=False)
                
                ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]), 'Importancia': importancia_variables}).sort_values('Importancia', ascending=False)
                
                pred = {}
                for i in datos_sin_clase.columns.values:
                    pred[i] = request.form[i]
                pred = pd.DataFrame([pred])
                prediccion = PronosticoAD.predict(pred)
                
                return render_template('home/resultado_pronostico.html', score=score, criterion=criterion, importancia_variables=importancia_variables, mae=mae, mse=mse, rmse=rmse, importancia_df=ImportanciaMod2.to_html(), prediccion=prediccion)
                
        if clas_option1 == "Clasificación":
            clas_option = request.form['modo_entrenamiento']
            
            if clas_option == "Modo 1 con random_state=0":
                ClasificacionAD = DecisionTreeClassifier(random_state=0)
                ClasificacionAD.fit(X_train, Y_train)
                Y_Clasificacion = ClasificacionAD.predict(X_test)
                
                accuracy = accuracy_score(Y_test, Y_Clasificacion) * 100
                criterion = ClasificacionAD.criterion
                importancia_variables = ClasificacionAD.feature_importances_
                classification = classification_report(Y_test, Y_Clasificacion)
                confusion = confusion_matrix(Y_test, Y_Clasificacion)
                
                ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]), 'Importancia': importancia_variables}).sort_values('Importancia', ascending=False)
                
                pred = {}
                for i in datos_sin_clase.columns.values:
                    pred[i] = request.form[i]
                pred = pd.DataFrame([pred])
                prediccion = ClasificacionAD.predict(pred)
                
                return render_template('home/resultado_clasificacion.html', accuracy=accuracy, criterion=criterion, importancia_variables=importancia_variables, classification=classification, confusion=confusion, importancia_df=ImportanciaMod2.to_html(), prediccion=prediccion)
                
            if clas_option == "Modo 2 con parámetros de número hojas, muestras y niveles del árbol":
                maxdepth = int(request.form['max_depth'])
                split = int(request.form['split'])
                leaf = int(request.form['leaf'])
                random = int(request.form['random_state'])
                
                ClasificacionAD = DecisionTreeClassifier(max_depth=maxdepth, min_samples_split=split, min_samples_leaf=leaf, random_state=random)
                ClasificacionAD.fit(X_train, Y_train)
                Y_Clasificacion = ClasificacionAD.predict(X_test)
                
                accuracy = accuracy_score(Y_test, Y_Clasificacion) * 100
                criterion = ClasificacionAD.criterion
                importancia_variables = ClasificacionAD.feature_importances_
                classification = classification_report(Y_test, Y_Clasificacion)
                confusion = confusion_matrix(Y_test, Y_Clasificacion)
                
                ImportanciaMod2 = pd.DataFrame({'Variable': list(datos_sin_clase[datos_sin_clase.columns.values]), 'Importancia': importancia_variables}).sort_values('Importancia', ascending=False)
                
                pred = {}
                for i in datos_sin_clase.columns.values:
                    pred[i] = request.form[i]
                pred = pd.DataFrame([pred])
                prediccion = ClasificacionAD.predict(pred)
                
                return render_template('home/resultado_clasificacion.html', accuracy=accuracy, criterion=criterion, importancia_variables=importancia_variables, classification=classification, confusion=confusion, importancia_df=ImportanciaMod2.to_html(), prediccion=prediccion)

