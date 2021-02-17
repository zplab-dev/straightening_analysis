import pathlib
import pickle
import freeimage
import collections
import numpy
import scipy
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from zplib.scalar_stats import kde
from elegant import load_data
from elegant import worm_spline
import image_utils