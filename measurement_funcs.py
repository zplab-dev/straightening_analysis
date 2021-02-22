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

def measure_area(worm1, worm1_mask, worm2, worm2_mask):
    """Compare the overall area differences between 2 worms. 
    NOTE: to extract these measurements use the extract_slice_area_measurements, or extract_area_measurements
    """
    worm1_pixels = worm1_mask.sum()
    worm2_pixels = worm2_mask.sum()

    diff = abs(worm1_pixels.astype(numpy.float64) - worm2_pixels.astype(numpy.float64))
    return diff, worm1_pixels.astype(numpy.float64)

def measure_integrated_gfp(worm1, worm1_mask, worm2, worm2_mask):
    """Measure the differences in integrated GFP pixels between two worms
    NOTE: to extract these measurements use the extract_slice_area_measurements, or extract_area_measurements
    """
    worm1_pixels = worm1[worm1_mask].sum()
    worm2_pixels = worm2[worm2_mask].sum()

    diff = abs(worm1_pixels.astype(numpy.float64) - worm2_pixels.astype(numpy.float64))
    return diff, worm1_pixels.astype(numpy.float64)

def measure_emd(worm1, worm1_mask, worm2, worm2_mask):
    """Evaluate the histograms between 2 worm images. We use the Earth Mover's Distance (EMD or Wasserstein distance)
    and the chi square test to evaluate the differences between the two histograms.

    Parameters:
        worm1 and worm2: numpy array of the worm images (bright field or gfp)
        worm1_mask and worm2_mask: binary array of the worm mask
    
    Returns:
        emd: earth mover's distance of the two images
    """
    worm1_pixels = worm1[worm1_mask]
    worm2_pixels = worm2[worm2_mask]
    worm1_hist, worm1_bins = numpy.histogram(worm1_pixels, bins=numpy.arange(0,2**16+1))
    worm2_hist, worm2_bins = numpy.histogram(worm2_pixels, bins=numpy.arange(0,2**16+1)) 
    #normalize the historgrams for area
    worm1_hist = worm1_hist/worm1_mask.sum()
    worm2_hist = worm2_hist/worm2_mask.sum()
      
    #earth mover's distance to see the difference between the two histograms
    bins = numpy.arange(0,2**16) #make bins that go from 0 to the max for uint16 images
    emd = scipy.stats.wasserstein_distance(bins, bins, worm1_hist, worm2_hist)

    return emd

def measure_chi_square(worm1, worm1_mask, worm2, worm2_mask):
    """Evaluate the histograms using chi squared as the metric
    """
    worm1_pixels = worm1[worm1_mask]
    worm2_pixels = worm2[worm2_mask]
    worm1_hist, worm1_bins = numpy.histogram(worm1_pixels, bins=numpy.arange(0,2**16+1))
    worm2_hist, worm2_bins = numpy.histogram(worm2_pixels, bins=numpy.arange(0,2**16+1)) 
    #normalize the historgrams for area
    worm1_hist = worm1_hist/worm1_mask.sum()
    worm2_hist = worm2_hist/worm2_mask.sum()
    #do chi-squared test
    chi = scipy.stats.chisquare((worm1_hist+1), f_exp=(worm2_hist+1))
    return chi[0] #the p-value doesn't really matter to us in this case