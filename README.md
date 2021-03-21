# custom-anomaly-detectors

A tiny package containing some custom anomaly detectors. In the <code>models.py</code> source code there are 4 detectors:

1. Multivariate Gaussian based anomaly detector

    <code>from pyad.models import MultivariateGaussian</code>

2. Multivariate T Distribution based anomaly detector

    <code>from pyad.models import MultivariateT</code>

3. Simple independence anomaly detector

    <code>from pyad.models import SimpleAnomalyDetector</code>

4. Influenced Outlierness based anomaly detector

    <code>from pyad.models import INFLO</code>

<h1>Dependencies</h1>

The package should run in any Conda Python 3.3+ environment with numpy, scipy and scikit-learn. Although I would recommend running in a Python 3.6+ environment.
However, you can pip install these packages by using the command line in this directory, and then run:

<code>conda create -n YOURENVNAME python=3.6</code>

<code>conda activate YOURENVNAME</code>

<code>pip install -r requirements.txt</code>

Link to Kaggle Credit Card Fraud Detection dataset:
https://www.kaggle.com/mlg-ulb/creditcardfraud