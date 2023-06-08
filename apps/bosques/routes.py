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


# Ruta para realizar el clustering
@blueprint.route('/bosques', methods=['POST'])
@login_required
def clustering():
    try:
        current_app.config["UPLOAD_FOLDER"] = "static/"
        if request.method == 'POST':
            f = request.files['file']
            filename = secure_filename(f.filename)
            basedir = os.path.abspath(os.path.dirname(__file__))
            f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
            # Obtener el archivo CSV enviado desde el formulario
            datos = pd.read_csv(filepath, header=None)
            # Procesamiento y entrenamiento del modelo
            X = datos.iloc[:, :-1].values
            y = datos.iloc[:, -1].values

            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42)

            tipo_bosques = request.form.get('algorithm')  # Obtener el tipo de bosques seleccionado

            if tipo_bosques == 'random_forest':
                modelo = RandomForestRegressor()
            elif tipo_bosques == 'multiple_trees':
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

            return render_template('bosque_result.html', mse=mse, mae=mae, r2=r2, plot_filename=plot_filename, filename='decision_tree.png')

    except Exception as e:
        return f"Error: {e}"
