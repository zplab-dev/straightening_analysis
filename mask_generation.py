import pathlib
import pickle
import freeimage
import collections
import numpy
import scipy
import sys
import random
import pkg_resources
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from zplib.scalar_stats import kde
from zplib.curve import interpolate
from elegant import load_data
from elegant import worm_spline

with pkg_resources.resource_stream('elegant', 'width_data/width_trends.pickle') as f:
    trend_data = pickle.load(f)
    WIDTH_TRENDS = trend_data

def to_tck(widths):
    x = numpy.linspace(0, 1, len(widths))
    smoothing = 0.0625 * len(widths)
    return interpolate.fit_nonparametric_spline(x, widths, smoothing=smoothing)

AVG_WIDTHS = numpy.array([numpy.interp(5, WIDTH_TRENDS['ages'], wt) for wt in WIDTH_TRENDS['width_trends']])
AVG_WIDTHS_TCK = to_tck(AVG_WIDTHS)
AVG_WIDTHS_TCK = (AVG_WIDTHS_TCK[0], AVG_WIDTHS_TCK[1], AVG_WIDTHS_TCK[2])


def unit_worm_tck(center_tck, keypoints, width_tck=AVG_WIDTHS_TCK, t_out=[0.0758344, 0.13659436, 0.544621766, 0.934377251]):
    """Generate a longitudinally 
    t_out default is average keypoint percentages across the worm
    """
    length = zplib.curve.spline_geometry.arc_length(center_tck)
    xs = numpy.array(list(keypoints.values()))[:,0]
    t_in = xs/length
    return worm_spline.longitudinal_warp_spline(t_in, t_out, center_tck, width_tck=width_tck)

def warp_unit_worm(lab_frame_image, center_tck, width_tck, keypoints, flip_vulva=True, mean_norm=False, t_out=[0.0758344, 0.13659436, 0.544621766, 0.934377251]):
    unit_center_tck, unit_width_tck = unit_worm_tck(center_tck, keypoints, width_tck=width_tck, t_out=t_out)
    worm_frame_image = worm_spline.to_worm_frame(lab_frame_image, unit_center_tck, unit_width_tck, standard_length=808, standard_width=AVG_WIDTHS_TCK)
    worm_frame_mask = worm_spline.worm_frame_mask(AVG_WIDTHS_TCK, worm_frame_image.shape)
    worm_frame_mask = worm_frame_mask>0
    if mean_norm:
        worm_frame_image = mean_normalize_image(worm_frame_image, worm_frame_mask)
    worm_frame_image[~worm_frame_mask]=0
    
    #account for vulva's being on different sides. I chose to put all vulvas pointing up
    if flip_vulva:
        if keypoints['vulva'][1]>0:
            worm_frame_image = numpy.flip(worm_frame_image, axis=1)


    return worm_frame_image

def lab_to_lab(lab_frame_image, center_tck, width_tck):
    """Generate an image of the full transform (lab -> worm -> lab).
    This is supposed to help us prove that warping doesn't really do anything
    to the resolution of the image
    """
    lab_mask = worm_spline.lab_frame_mask(center_tck, width_tck, lab_frame_image.shape)
    lab_frame_mask = lab_mask>0

    #go lab->worm->lab
    worm_frame_image = worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck=width_tck)
    worm_to_lab_image = worm_spline.to_lab_frame(worm_frame_image, lab_frame_image.shape, center_tck, width_tck)

    worm_mask = worm_spline.lab_frame_mask(center_tck, width_tck, worm_to_lab_image.shape)
    worm_frame_mask = worm_mask>0

    return (lab_frame_mask, worm_to_lab_image, worm_frame_mask)

def generate_worm_masks(lab_frame_image, center_tck, width_tck):
    #make a lab frame mask
    lab_mask = worm_spline.lab_frame_mask(center_tck, width_tck, lab_frame_image.shape)
    lab_frame_mask = lab_mask>0

    #get worm_frame image/mask
    worm_frame_image = worm_spline.to_worm_frame(lab_frame_image, center_tck, width_tck=width_tck)
    worm_mask = worm_spline.worm_frame_mask(width_tck, worm_frame_image.shape)
    worm_frame_mask = worm_mask>0

    return ([lab_frame_mask], [worm_frame_image], [worm_frame_mask])

def generate_longitudinal_worm_masks(lab_frame_image, center_tck, width_tck, keypoints):
    """Generate worm masks for a worm and longitudinally straigthen them
    """
    lab_mask = worm_spline.lab_frame_mask(center_tck, width_tck, lab_frame_image.shape)
    lab_frame_mask = lab_mask>0
    worm_frame_image = warp_unit_worm(lab_frame_image, center_tck, width_tck, keypoints, flip_vulva=False, t_out=[0.0758344, 0.13659436, 0.544621766, 0.934377251])
    worm_mask = worm_spline.worm_frame_mask(AVG_WIDTHS_TCK, worm_frame_image.shape)
    worm_frame_mask = worm_mask>0

    return ([lab_frame_mask], [worm_frame_image], [worm_frame_mask])

def generate_longitudinal_checkerboard_masks(lab_frame_image, center_tck, width_tck, keypoints):
    lab_frame_mask, worm_frame_image, worm_frame_mask = generate_longitudinal_worm_masks(lab_frame_image, center_tck, width_tck, keypoints)
    lab_frame_mask, worm_frame_image, worm_mask = lab_frame_mask[0], worm_frame_image[0], worm_frame_mask[0] 
    new_worm_frame_images, worm_frame_masks = [],[]
    long_tck, long_width_tck = unit_worm_tck(center_tck, keypoints)

    for start in numpy.arange(0,1,0.1):
        length = worm_frame_image.shape[0]
        end =  start+0.2
        
        width = worm_frame_image.shape[1]
        x,y = numpy.indices(worm_frame_image.shape)
        x_slice = (x>=(length*start)) & (x<=(length*end))

        #top part of checkerboard
        y_slice = y>=int(width/2)
        top_mask = worm_mask & y_slice & x_slice
        top_mask = top_mask>0
        worm_frame_masks.append(top_mask)
        new_worm_frame_images.append(worm_frame_image)

        #bottom part of checkerboard
        y_slice = y<int(width/2)
        bot_mask = worm_mask & y_slice & x_slice
        bot_mask = bot_mask>0
        worm_frame_masks.append(bot_mask)
        new_worm_frame_images.append(worm_frame_image)

    lab_frame_masks = worm_spline.to_lab_frame([wm.astype(int) for wm in worm_frame_masks], lab_frame_image.shape, long_tck, width_tck, order=1)
    lab_frame_masks = [lm>0 for lm in lab_frame_masks]
    #lab_frame_masks = [1]
    return (lab_frame_masks, new_worm_frame_images, worm_frame_masks)

def generate_checkerboard_slice_masks(lab_frame_image, center_tck, width_tck):
    """Generate masks for the lab and worm fromes in sections of the worm (slices), but checkerboarded above and below
    the centerline. This is to give us a little better resolution than the whole slices (especially for area measurements)
    and see how many pixels are being created/destroyed between warping and unwarping
    """
    lab_frame_slices, worm_frame_images, worm_frame_slices = generate_sliced_masks_fast(lab_frame_image, center_tck, width_tck)
    new_worm_frame_images, worm_frame_masks = [],[]
    
    for worm_image, worm_mask in zip(worm_frame_images, worm_frame_slices):
        x, y = numpy.indices(worm_image.shape)
        width = worm_image.shape[1]
        
        #top part of checkerboard
        y_slice = y>int(width/2)
        top_mask = worm_mask & y_slice
        top_mask = top_mask>0
        worm_frame_masks.append(top_mask)
        worm_frame_images.append(worm_image)

        #bottom part of checkerboard
        y_slice = y<int(width/2)
        bot_mask = worm_mask & y_slice
        bot_mask = bot_mask>0
        worm_frame_masks.append(bot_mask)
        worm_frame_images.append(worm_image)

    lab_frame_masks = worm_spline.to_lab_frame([wm.astype(int) for wm in worm_frame_masks], lab_frame_image.shape, center_tck, width_tck, order=1)
    lab_frame_masks = [lm>0 for lm in lab_frame_masks]
    
    return (lab_frame_masks, worm_frame_images, worm_frame_masks)

def generate_sliced_masks_fast(lab_frame_image, center_tck, width_tck):
    lab_frame_mask, worm_frame_image, worm_frame_mask = generate_worm_masks(lab_frame_image, center_tck, width_tck)
    lab_frame_mask, worm_frame_image, worm_frame_mask = lab_frame_mask[0], worm_frame_image[0], worm_frame_mask[0]

    worm_frame_images, worm_frame_masks = [], []

    for start in numpy.arange(0,1,0.1):
        end = start+0.2
        length = worm_frame_image.shape[0]

        x,y = numpy.indices(worm_frame_image.shape)
        x_slice = (x>=start*length) & (x<=end*length)
        new_worm_frame_mask = x_slice & worm_frame_mask
        new_worm_frame_mask = new_worm_frame_mask>0
        worm_frame_masks.append(new_worm_frame_mask)
        worm_frame_images.append(worm_frame_image)

    new_lab_frame_masks = worm_spline.to_lab_frame(worm_frame_masks, lab_frame_image.shape, center_tck, width_tck, order=0)
    lab_frame_masks = [lm>0 for lm in new_lab_frame_masks]

    return(lab_frame_masks, worm_frame_images, worm_frame_masks)

def generate_sliced_masks(lab_frame_image, center_tck, width_tck):
    """Generate masks for the lab and worm frames in sections of the worm (slices). This will be used for diving deeper into
    if we are losing information between warped and unwarped worms. We will see if warping truly alters the fluorescence and how
    much it does that. This slices based on percentage of length, not any keypoints.

    Parameters:
        lab_frame_image: image from the lab frame that you want to straighten
        center_tck:
        width_tck:

    """

    lab_frame_mask, worm_frame_image, worm_frame_mask = generate_worm_masks(lab_frame_image, center_tck, width_tck)
    lab_frame_mask, worm_frame_image, worm_frame_mask = lab_frame_mask[0], worm_frame_image[0], worm_frame_mask[0]
    #slice n dice (currently only have 5 regions)
    worm_frame_images, worm_frame_masks = [], []
    for start in numpy.arange(0,1,0.1):
        end = start + 0.2
        length = worm_frame_image.shape[0]
        mask_slice = numpy.zeros(worm_frame_mask.shape)
        mask_slice[int(length*start):int(length*end),] = 1 #make slices in the worm frame
        mask_slice = mask_slice>0
        new_worm_frame_mask = mask_slice&worm_frame_mask
        worm_frame_masks.append(new_worm_frame_mask)
        worm_frame_images.append(worm_frame_image)

    new_lab_frame_masks = worm_spline.to_lab_frame(worm_frame_masks, lab_frame_image.shape, center_tck, width_tck, order=0)
    lab_frame_masks = [lm>0 for lm in new_lab_frame_masks]
    
    return (lab_frame_masks, worm_frame_images, worm_frame_masks)

def slice_mask(warp_image, mask, slices):
    """ Measure only a section of the worm.

    Parameters:
        warp_image: numpy array with the image data for the warped worm
        mask: numpy array of the warped worm mask 
            NOTE: warp_image() gives the warp_image and the mask as a tuple
        slice: tuple with the slice you want to take along the length of the worm.
        The tuple will contain percentages of where you want to start and end
            (i.e. if you wanted the first 10% of the worm you would use:
                slice_worms(warp_image, mask, (0,10))

    Returns:
        Tuple of the slice of the warp_image and
        slice of the mask in the form (image_slice, warp_slice)
        
        image_slice: numpy array of the sliced image of the warp_image

        mask: numpy array of the sliced mask
    """

    length = warp_image.shape[0]
    start_per, end_per = slices

    start = start_per/100
    end = end_per/100

    #get the slices from the image
    #Note: to get a percentage along the backbone we need to multiply
    #the length by the start and end percentages
    mask[int(length*start):int(length*end),] = 0

    return(mask)