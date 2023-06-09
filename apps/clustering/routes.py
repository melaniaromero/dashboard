from flask import Flask, render_template, request, current_app
from apps.clustering import blueprint
from flask_login import login_required
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from werkzeug.utils import secure_filename
import os
from jinja2 import TemplateNotFound
import io
import base64

# Crear una aplicación Flask
app = Flask(__name__)

# Ruta de inicio
@blueprint.route('/clustering')
@login_required
def index():
    return render_template('home/clustering.html')

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

@blueprint.route('/clustering', methods=['POST'])
@login_required
def clustering():
    num_clusters = int(request.form['num_clusters'])
    try:
        current_app.config["UPLOAD_FOLDER"] = "static/"
        if request.method == 'POST':
            f = request.files['file']
            clustering_type = request.form.get('clustering_type')  # Obtener el tipo de clustering seleccionado
            distance_metric = request.form.get('distance_metric')  # Obtener la métrica de distancia seleccionada

            filename = secure_filename(f.filename)

            basedir = os.path.abspath(os.path.dirname(__file__))

            f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)

            # Obtener el archivo CSV enviado desde el formulario
            df = pd.read_csv(filepath, header=None)

            # Eliminar las primeras filas no numéricas del DataFrame
            df = df.apply(pd.to_numeric, errors='coerce').dropna()

            if df.shape[0] < 1:
                return "Error: No se encontraron datos numéricos después de la eliminación de filas no numéricas."

            # Preprocesamiento de los datos
            estandarizar = StandardScaler()
            MEstandarizada = estandarizar.fit_transform(df)
            elbow_plot_path = ''  # Definir elbow_plot_path con un valor predeterminado

            if clustering_type == 'particional':
                # Clustering particional (K-means)
                MKMeans = KMeans(n_clusters=num_clusters, random_state=0)

                # Determinar el número óptimo de clusters usando el método del codo
                num_clusters_range = range(1, 11)
                inertia = []
                for k in num_clusters_range:
                    MKMeans.n_clusters = k
                    MKMeans.fit(MEstandarizada)
                    inertia.append(MKMeans.inertia_)

                # Gráfico del codo
                plt.figure(figsize=(8, 6))
                plt.plot(num_clusters_range, inertia, marker='o', linestyle='-', color='b')
                plt.xlabel('Número de clusters')
                plt.ylabel('Inercia')
                plt.title('Método del Codo')
                elbow_plot_path = os.path.join('static', 'elbow_plot.png')
                plt.savefig(os.path.join(basedir, elbow_plot_path))
                plt.close()
                elbow_plot_base64 = convert_image_to_base64(elbow_plot_path)

                # Gráfico de puntos distribuidos estandarizados
                plt.figure(figsize=(8, 6))
                plt.scatter(MEstandarizada[:, 0], MEstandarizada[:, 1], c=MKMeans.labels_, cmap='viridis')
                plt.xlabel(df.columns[0])
                plt.ylabel(df.columns[1])
                plt.title('Puntos Distribuidos Estandarizados')
                scatter_plot_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'scatter_plot.png')
                plt.savefig(scatter_plot_path)
                plt.close()
                scatter_plot_base64 = convert_image_to_base64(scatter_plot_path)

                # Obtener los resultados finales con el número óptimo de clusters
                MKMeans.n_clusters = num_clusters
                MKMeans_labels = MKMeans.fit_predict(MEstandarizada)

                # Agregar las etiquetas de clúster al DataFrame para el clustering particional
                df['cluster'] = MKMeans_labels

                # Calcular los centroides de los clústeres para el clustering particional
                Centroides = df.groupby(['cluster']).mean()

                # Pasar los resultados a la plantilla HTML correspondiente
                return render_template('home/clustering_par.html', scatter_plot_base64=scatter_plot_base64,
                                       elbow_plot_base64=elbow_plot_base64, centroids=Centroides.to_html())

            elif clustering_type == 'jerarquico':
                if distance_metric == 'manhattan':
                    MJerarquico = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete', affinity='l1')
                elif distance_metric == 'chebyshev':
                    MJerarquico = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete', affinity='chebyshev')
                elif distance_metric == 'euclidean':
                    MJerarquico = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete', affinity='euclidean')
                elif distance_metric == 'minkowski':
                    MJerarquico = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete', affinity='minkowski', p=2)
                else:
                    return "Error: Métrica de distancia no válida."
                MJerarquico_labels = MJerarquico.fit_predict(MEstandarizada)

                # Agregar las etiquetas de clúster al DataFrame para el clustering jerárquico
                df['cluster'] = MJerarquico_labels

                # Calcular los centroides de los clústeres para el clustering jerárquico
                Centroides = df.groupby(['cluster']).mean()

                # Crear el gráfico de dispersión para el clustering jerárquico
                plt.figure(figsize=(8, 6))
                plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['cluster'], cmap='viridis')
                plt.xlabel(df.columns[0])
                plt.ylabel(df.columns[1])
                plt.title('Cluster Scatter Plot (Jerárquico)')
                plt.colorbar(label='Cluster')
                scatter_plot_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'scatter_plot.png')
                plt.savefig(scatter_plot_path)
                plt.close()
                scatter_plot_base64 = convert_image_to_base64(scatter_plot_path)

                # Crear el mapa de calor (heatmap) para el clustering jerárquico
                plt.figure(figsize=(8, 6))
                correlation_matrix = df.corr()
                mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
                sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='viridis')
                plt.xlabel(df.columns[0])
                plt.ylabel(df.columns[1])
                plt.title('Cluster Heatmap (Jerárquico)')
                heatmap_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'heatmap.png')
                plt.savefig(heatmap_path)
                plt.close()
                heatmap_path_base64 = convert_image_to_base64(heatmap_path)

                # Crear el pairplot para el clustering jerárquico
                plt.figure(figsize=(8, 6))
                sns.pairplot(df, vars=df.columns[:2], hue='cluster', palette='viridis')
                plt.title('Pair Plot (Jerárquico)')
                pairplot_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'pairplot.png')
                plt.savefig(pairplot_path)
                plt.close()
                pairplot_path_base64 = convert_image_to_base64(pairplot_path)

                # Crear el dendrograma
                Z = linkage(MEstandarizada, method='complete', metric=distance_metric)
                plt.figure(figsize=(12, 8))
                dendrogram(Z)
                plt.xlabel('Samples')
                plt.ylabel('Distance')
                plt.title('Dendrogram')
                dendrogram_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'dendrogram.png')
                plt.savefig(dendrogram_path)
                plt.close()
                dendrogram_path_base64 = convert_image_to_base64(dendrogram_path)

                # Pasar los resultados a la plantilla HTML correspondiente
                return render_template('home/clustering_jer.html', scatter_plot_base64=scatter_plot_base64,
                                       dendrogram_path_base64=dendrogram_path_base64, heatmap_path_base64=heatmap_path_base64,
                                       pairplot_path_base64=pairplot_path_base64, centroids=Centroides.to_html())
    except Exception as e:
        return f"Error: {e}"

