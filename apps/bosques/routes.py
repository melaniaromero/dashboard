# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
from flask import Flask, render_template, request, current_app
from flask_login import login_required
from jinja2 import TemplateNotFound
from werkzeug.utils import secure_filename
from apps.clustering import blueprint
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

app = Flask(__name__)

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

# Ruta para realizar el clustering
@blueprint.route('/bosques', methods=['POST'])
@login_required
def bosques():
    current_app.config["UPLOAD_FOLDER"] = "static/"
    if 
    if request.method == 'POST':
        	f = request.files['file']
            
            filename = secure_filename(f.filename)

			basedir = os.path.abspath(os.path.dirname(__file__))

			f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
			filepath=os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
						
			# Obtener el archivo CSV enviado desde el formulario
			df = pd.read_csv(filepath, header=None)
        archivo_bosque_1 = request.files['archivo']
        datos = pd.read_csv(archivo_bosque_1)
        
        # Procesamiento y entrenamiento del modelo
        X = datos.iloc[:, :-1].values
        y = datos.iloc[:, -1].values
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
        
        tipo_bosques = request.form.get('tipo_bosques')  # Obtener el tipo de bosques seleccionado
        
        if tipo_bosques == 'aleatorios':
            modelo = RandomForestRegressor()
        elif tipo_bosques == 'multiples':
            modelo = RandomForestClassifier()
        else:
            modelo = DecisionTreeClassifier()
        
        modelo.fit(X_train, y_train)
        
        y_pred = modelo.predict(X_test)
        
        # Evaluación del modelo
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Visualización de resultados
        plt.scatter(y_test, y_pred)
        plt.xlabel('Valores Reales')
        plt.ylabel('Valores Predichos')
        plt.title('Predicciones de Bosques')
        plot_filename = 'plot.png'  # Nombre de archivo de la imagen
        plt.savefig('static/' + plot_filename)  # Guardar la imagen en la carpeta "static"
        plt.close()
        
        if isinstance(modelo, RandomForestClassifier):
            class_names = np.unique(y)  # Definir los nombres de las clases
            # Obtener el primer árbol del bosque aleatorio
            tree = modelo.estimators_[0]
            
            dot_data = export_graphviz(tree, out_file=None,
                                    feature_names=X_train.columns.values,
                                    class_names=class_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
            
            graph = graphviz.Source(dot_data)
            graph.render(filename='decision_tree', format='png', directory=os.path.join(app.root_path, 'static'), cleanup=True)

        return render_template('bosques_res.html', mse=mse, mae=mae, r2=r2, plot_filename=plot_filename, filename='decision_tree.png')

    return render_template('bosques.html')

if __name__ == '__main__':
    app.run(debug=True)
