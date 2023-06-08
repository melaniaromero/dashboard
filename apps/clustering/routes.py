# Importar las bibliotecas necesarias
from flask import Flask, render_template, request, current_app
from apps.clustering import blueprint
from flask_login import login_required

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from werkzeug.utils import secure_filename
import os
from jinja2 import TemplateNotFound


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


# Ruta para realizar el clustering
@blueprint.route('/clustering', methods=['POST'])
@login_required
def clustering():
	try:
		current_app.config["UPLOAD_FOLDER"] = "static/"
		if request.method == 'POST':
			f = request.files['file']

			filename = secure_filename(f.filename)

			basedir = os.path.abspath(os.path.dirname(__file__))

			f.save(os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename))
			filepath=os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], filename)
						
			# Obtener el archivo CSV enviado desde el formulario
			df = pd.read_csv(filepath, header=None)

			# Eliminar las primeras filas no numéricas del DataFrame
			df = df.apply(pd.to_numeric, errors='coerce').dropna()

			if df.shape[0] < 1:
				return "Error: No se encontraron datos numéricos después de la eliminación de filas no numéricas."

			# Preprocesamiento de los datos
			estandarizar = StandardScaler()
			MEstandarizada = estandarizar.fit_transform(df)

			# Clustering jerárquico
			MJerarquico = AgglomerativeClustering(n_clusters=4, linkage='complete', affinity='euclidean')
			MJerarquico_labels = MJerarquico.fit_predict(MEstandarizada)

			# Agregar las etiquetas de clúster al DataFrame
			df['clusterH'] = MJerarquico_labels

			# Calcular los centroides de los clústeres
			CentroidesH = df.groupby(['clusterH']).mean()

			# Crear el gráfico de dispersión
			plt.figure(figsize=(8, 6))
			plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['clusterH'], cmap='viridis')
			plt.xlabel(df.columns[0])
			plt.ylabel(df.columns[1])
			plt.title('Cluster Scatter Plot')
			plt.colorbar(label='Cluster')
			scatter_plot_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'scatter_plot.png')
			plt.savefig(scatter_plot_path)
			plt.close()

			# Crear el dendrograma
			Z = linkage(MEstandarizada, method='complete', metric='euclidean')
			plt.figure(figsize=(12, 8))
			dendrogram(Z)
			plt.xlabel('Samples')
			plt.ylabel('Distance')
			plt.title('Dendrogram')
			dendrogram_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'dendrogram.png')
			plt.savefig(dendrogram_path)
			plt.close()

			# Crear el mapa de calor
			plt.figure(figsize=(10, 8))
			sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
			plt.title('Correlation Heatmap')
			heatmap_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'heatmap.png')
			plt.savefig(heatmap_path)
			plt.close()

			# Crear el pairplot
			sns.set(style="ticks")
			pairplot = sns.pairplot(df, hue='clusterH')
			pairplot_path = os.path.join(basedir, current_app.config['UPLOAD_FOLDER'], 'pairplot.png')
			pairplot.savefig(pairplot_path)
			plt.close()

			# Pasar los resultados a la plantilla HTML
			return render_template('home/clustering_results.html', scatter_plot_path=scatter_plot_path, dendrogram_path=dendrogram_path, heatmap_path=heatmap_path, pairplot_path=pairplot_path, centroids=CentroidesH.to_html())

	except Exception as e:
		return f"Error: {e}"
	
