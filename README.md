# Wetsuit

[![image](https://img.shields.io/badge/python-3.7--3.11-blue.svg)](https://www.python.org)
[![image](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

A Scikit-Learn wrapper for H2O Estimators.

## Why Wetsuit

While H2O Estimators have the `.fit()` and `.predict()` methods of the Scikit-Learn API, they don't always
function as expected, especially with `Pipeline` objects. This package contains two estimators and a
single transformer to remedy.

For example. the `H2OEstimator.fit()` method expects two `H2OFrame` objects, vice pandas `DataFrame` or
numpy `NDArray` objects. Wetsuit provides two options for handling this behavior:

- `WetsuitRegressor` and `WetsuitClassifier` classes that wrap `H2OEstimator` objects and handle type conversion automatically, within the `.fit()` and `.predict()` methods.
- `H2oFrameTransformer` class that converts both `DataFrame` and `NDArray` objects to `H2OFrame` objects via `.fit_transform()`, and an `.inverse_transform()` method to convert back.

## Install

Create a virtual environment with Python >= 3.7 and install from PyPI:

```bash
pip install wetsuit
```

## Use

### Basic Pipeline

Here's an example that shows Scikit-Learn `Pipeline` compatibility. To align with the H2O API,
we must instantiate the `WetsuitClassifier` with a list of feature names and the name of the
response variable (these can also be indices). From there, you can plug in to a basic `Pipeline`
object.

```python
import h2o
from h2o.estimators import H2OXGBoostEstimator
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import wetsuit

h2o.init()

data = load_iris()

cls = wetsuit.WetsuitClassifier(H2OXGBoostEstimator(), data['feature_names'], 'target')
pl = Pipeline([
    ('scaler', StandardScaler()),
    ('cls', cls)
])
pl.fit(data['data'], data['target'])
fitted = pl.predict(data['data'])

h2o.cluster().shutdown()
```

*Note*: If you're doing feature selection within the pipeline, it's best instantiate the `WetsuitClassifier`
from within the pipeline, so that you can dynamically pass a list of selected features using a
selector's `.get_feature_names_out()` method.


## Documentation

Documentation hosted on Github Pages: [https://chris-santiago.github.io/wetsuit/](https://chris-santiago.github.io/wetsuit/)
