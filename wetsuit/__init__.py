"""Package initialization."""
from importlib.metadata import version

from wetsuit.models import WetsuitClassifier, WetsuitRegressor
from wetsuit.transforms import H2oFrameTransformer


__version__ = version("wetsuit")
