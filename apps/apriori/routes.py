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
matplotlib.use('Agg')

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


