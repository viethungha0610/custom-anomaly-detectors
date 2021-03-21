# custom-anomaly-detectors

<h1>Overview</h1>

A tiny package containing some custom anomaly detectors. In the <code>models.py</code> source code there are 4 detectors:

1. Multivariate Gaussian based anomaly detector

    <code>from pyad.models import MultivariateGaussian</code>

2. Multivariate T Distribution based anomaly detector

    <code>from pyad.models import MultivariateT</code>

3. Simple independence anomaly detector

    <code>from pyad.models import SimpleAnomalyDetector</code>

4. Influenced Outlierness based anomaly detector

    <code>from pyad.models import INFLO</code>

<h1>Installation and Dependencies</h1>

This is 'almost' a Python package (I will continuously work on this). But to use the code right away, simply clone this Github repo:

<code>git clone https://github.com/viethungha0610/custom-anomaly-detectors.git</code>

<h4>Environment</h4>

No complicated dependencies here. The package should run in any basic Conda Python 3.3+ environment with numpy, scipy and scikit-learn. Although I would recommend running in a Python 3.6+ environment.

However, you can pip install these packages by using the command line in this directory, and then run:

<code>conda create -n YOURENVNAME python=3.6</code>

<code>conda activate YOURENVNAME</code>

<code>pip install -r requirements.txt</code>

<h4>Using the code</h4>

Open a Jupyter notebook, import the classes from the first section and get going right away.

<h1>Suggested test dataset</h1>

Link to Kaggle Credit Card Fraud Detection dataset:
https://www.kaggle.com/mlg-ulb/creditcardfraud