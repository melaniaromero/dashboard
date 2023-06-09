# Importar las bibliotecas necesarias
from flask import Flask, render_template, request, current_app
from apps.bosques import blueprint
from flask_login import login_required
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from werkzeug.utils import secure_filename
import os
from jinja2 import TemplateNotFound
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.tree import DecisionTreeRegressor
import pydotplus
from io import StringIO
from sklearn.metrics import roc_curve, auc
import io
import base64
import graphviz

# Ruta de inicio
@blueprint.route('/bosques')
@login_required
def index():
    return render_template('home/bosques.html')

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
    
def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

@blueprint.route('/bosques', methods=['POST'])
@login_required
def bosques():
    try:
        basedir = os.path.abspath(os.path.dirname(__file__))
        upload_folder = os.path.join(basedir, "static")
        current_app.config["UPLOAD_FOLDER"] = upload_folder
        
        if request.method == 'POST':
            f = request.files['file']
            filename = secure_filename(f.filename)
            basedir = os.path.abspath(os.path.dirname(__file__))
            f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
            # Obtener el archivo CSV enviado desde el formulario
            datos = pd.read_csv(filepath, header=None, skiprows=1, names=['feature1', 'feature2', 'feature3', 'target'])
            # Procesamiento y entrenamiento del modelo
            X = datos.iloc[:, :-1].values
            y = datos.iloc[:, -1].values
            # Obtener el tamaño de prueba ingresado por el usuario
            test_size = int(request.form.get('test_size'))
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size/100, random_state=42)

            tipo_bosques = request.form.get('algorithm')  # Obtener el tipo de bosques seleccionado

            if tipo_bosques == 'random_forest':
                modelo = RandomForestRegressor()
            elif tipo_bosques == 'multiple_trees':
                modelo = RandomForestRegressor()
            else:
                modelo = DecisionTreeRegressor()
                
            modelo.fit(X_train, y_train)
            
            y_pred = modelo.predict(X_test)


            modelo.fit(X_train, y_train)

            y_pred = modelo.predict(X_test)

            # Evaluación del modelo
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Obtener las probabilidades de la clase positiva
            if isinstance(modelo, RandomForestClassifier):
                proba = modelo.predict_proba(X_test)[:, 1]
            else:
                proba = modelo.predict(X_test)

            # Calcular la curva ROC
            fpr, tpr, _ = roc_curve(y_test, proba)
            roc_auc = auc(fpr, tpr)

            # Visualización de resultados
            plt.plot(fpr, tpr, label='Curva ROC (AUC = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend(loc="lower right")
            roc_filename = os.path.join(current_app.config['UPLOAD_FOLDER'], 'roc_curve.png')
            plt.savefig(os.path.join(basedir, roc_filename))  # Guardar la imagen en la carpeta "static"
            plt.close()
            roc_plot_base64 = convert_image_to_base64(roc_filename)

            plot_filename = os.path.join(current_app.config['UPLOAD_FOLDER'], 'plot.png')
            plot_base64 = convert_image_to_base64(plot_filename)

            tree_filename = None
            tree_plot_base64 = None

            if isinstance(modelo, RandomForestClassifier):
                class_names = np.unique(y)  # Definir los nombres de las clases
                # Obtener el primer árbol del bosque aleatorio
                tree = modelo.estimators_[0]

                plt.figure(figsize=(12, 6))
                export_graphviz(tree,
                                feature_names=datos.columns[:-1],
                                class_names=class_names,
                                filled=True, rounded=True,
                                special_characters=True)
                plt.savefig(os.path.join(upload_folder, 'decision_tree.png'))
                tree_filename = os.path.join(upload_folder, 'decision_tree.png')
                tree_plot_base64 = convert_image_to_base64(tree_filename)

            return render_template('home/bosques_res.html', mse=mse, mae=mae, r2=r2, plot_base64=plot_base64, tree_plot_base64=tree_plot_base64, roc_plot_base64=roc_plot_base64)

    except Exception as e:
        return f"Error: {e}"