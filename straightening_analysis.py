import pathlib
import pickle
import freeimage
import collections
import numpy
import scipy
import sys
import random
from elegant import preprocess_image

def preprocess_image(timepoint, img_type='bf'):
	position_root = timepoint.position.path
    image_corrected = preprocess_image.flatfield_correct(position_root, timepoint.name, img_type)
    return image_corrected

def measure_timepoint(timepoint, mask_generation, measurement_func, longitudinal_warp=False, img_type='bf'):
	tp1 = timepoint
	tp_idx = list(tp1.position.timepoints.keys()).index(tp1.name)
	tp2 = list(tp1.position.timepoints.values())[tp_idx+1]

	#get random worm
	random_position = random.choice(list(tp1.position.experiment.positions.keys()))
	tr = random.choice(list(tp.position.experiment.positions[random_worm].timepoints.values()))
	while tr == tp1 or tr == tp2:
		random_position = random.choice(list(tp1.position.experiment.positions.keys()))
		tr = random.choice(list(tp.position.experiment.positions[random_worm].timepoints.values()))

	lab_frame_image_t1 = preprocess_image(tp1)
    lab_frame_image_t2 = preprocess_image(tp2)
    lab_frame_image_tr = preprocess_image(tpr)

    center_tck_t1, width_tck_t1 = tp1.annotations['pose']
    center_tck_t2, width_tck_t2 = tp2.annotations['pose']
    center_tck_tr, width_tck_tr = tr.annotations['pose']

    #generate the masks for each worm
    measurements = []
    if longitudinal_warp:
        keypoints_t1 = annotations[worm_name][1][tp1]['keypoints']
        keypoints_t2 = annotations[worm_name][1][tp2]['keypoints']
        keypoints_tr = annotations[random_worm][1][random_time]['keypoints']

        lab_frame_masks_t1, worm_frame_images_t1, worm_frame_masks_t1 = mask_generation(lab_frame_image_t1, center_tck_t1, width_tck_t1, keypoints_t1)
        lab_frame_masks_t2, worm_frame_images_t2, worm_frame_masks_t2 = mask_generation(lab_frame_image_t2, center_tck_t2, width_tck_t2, keypoints_t2)
        lab_frame_masks_tr, worm_frame_images_tr, worm_frame_masks_tr = mask_generation(lab_frame_image_tr, center_tck_tr, width_tck_tr, keypoints_tr)

    else:
        lab_frame_masks_t1, worm_frame_images_t1, worm_frame_masks_t1 = mask_generation(lab_frame_image_t1, center_tck_t1, width_tck_t1)
        lab_frame_masks_t2, worm_frame_images_t2, worm_frame_masks_t2 = mask_generation(lab_frame_image_t2, center_tck_t2, width_tck_t2)
        lab_frame_masks_tr, worm_frame_images_tr, worm_frame_masks_tr = mask_generation(lab_frame_image_tr, center_tck_tr, width_tck_tr)
    
    for lab_frame_mask_t1, worm_frame_image_t1, worm_frame_mask_t1, lab_frame_mask_t2, lab_frame_mask_tr in zip(lab_frame_masks_t1, 
            worm_frame_images_t1, worm_frame_masks_t1, lab_frame_masks_t2, lab_frame_masks_tr):
        #evaluate the measurement of comparison between worm t and worm t+1 and lab_frame_t1 and worm_frame_t1
        compare_value_tt1 = measurement_func(lab_frame_image_t1, lab_frame_mask_t1, lab_frame_image_t2, lab_frame_mask_t2)
        #evaluate the measurement of comparison between unwarped worm t and its warped image
        compare_value_tt = measurement_func(lab_frame_image_t1, lab_frame_mask_t1, worm_frame_image_t1, worm_frame_mask_t1)
        #measurement of comparison between worm t and random worm
        compare_value_ttr = measurement_func(lab_frame_image_t1, lab_frame_mask_t1, lab_frame_image_tr, lab_frame_mask_tr)
        #lab_frame_image_t1[lab_frame_mask_t1] = 1

        measurements.append({'consecutive timepoint measurements': compare_value_tt1, 
            'warp to unwarped' : compare_value_tt, 
            'worm vs random worm': (compare_value_ttr, (random_worm, random_time))})

    return measurements  
	

def measure_position(position, mask_generation, measurement_func, longitudinal_warp=False):
    """Measure something for one worm using the mask_generation as the way to generate the mask

    Parameters:
        timepoint: Timepoint object for one timepoint
        mask_generation: A function that produces a lab frame mask, a worm frame image, and a worm frame mask.
            The function's signature must be: mask_generation(lab_frame_image, center_tck, width_tck) and return
            a tuple of lists of the masks ([lab_frame_masks], [worm_frame_images],[worm_frame_masks])
            NOTE:see lab_to_lab and generate_worm_masks for examples
        measurement_func: A function that returns a measurement of comparison between worms. Function signature must be:
            measurement_func(worm1_image, worm1_mask, worm2_image, worm2_mask)
        longitudinal_warp: Boolean describing whether or not the mask generation function requires the keypoints to work
    """
    timepoint_measurements = collections.OrderedDict()
    times = list(positions[worm_name].keys())
    
    for tp1, tp2 in zip(times, times[1:]):
        #get consecutive worm images (ie. worm t and worm t+1)
        lab_frame_image_t1 = normalize_gfp_image(positions[worm_name][tp1][0])
        lab_frame_image_t2 = normalize_gfp_image(positions[worm_name][tp2][0])
        #lab_frame_image_t1 = freeimage.read(positions[worm_name][tp1][0])
        #lab_frame_image_t2 = freeimage.read(positions[worm_name][tp2][0])

        center_tck_t1, width_tck_t1 = annotations[worm_name][1][tp1]['pose']
        center_tck_t2, width_tck_t2 = annotations[worm_name][1][tp2]['pose']

        #find a random worm to compare with t1
        random_worm = random.choice(list(positions.keys()))
        random_time = random.choice(list(positions[random_worm].keys()))
        if random_time is tp1 and random_worm is worm_name:
            random_worm = random.choice(list(positions.keys()))
            random_time = random.choice(list(positions[random_worm].keys()))
        lab_frame_image_tr = normalize_gfp_image(positions[random_worm][random_time][0])
        center_tck_tr, width_tck_tr = annotations[random_worm][1][random_time]['pose']

        #generate the masks for each worm
        measurements = []
        if longitudinal_warp:
            keypoints_t1 = annotations[worm_name][1][tp1]['keypoints']
            keypoints_t2 = annotations[worm_name][1][tp2]['keypoints']
            keypoints_tr = annotations[random_worm][1][random_time]['keypoints']

            lab_frame_masks_t1, worm_frame_images_t1, worm_frame_masks_t1 = mask_generation(lab_frame_image_t1, center_tck_t1, width_tck_t1, keypoints_t1)
            lab_frame_masks_t2, worm_frame_images_t2, worm_frame_masks_t2 = mask_generation(lab_frame_image_t2, center_tck_t2, width_tck_t2, keypoints_t2)
            lab_frame_masks_tr, worm_frame_images_tr, worm_frame_masks_tr = mask_generation(lab_frame_image_tr, center_tck_tr, width_tck_tr, keypoints_tr)

        else:
            lab_frame_masks_t1, worm_frame_images_t1, worm_frame_masks_t1 = mask_generation(lab_frame_image_t1, center_tck_t1, width_tck_t1)
            lab_frame_masks_t2, worm_frame_images_t2, worm_frame_masks_t2 = mask_generation(lab_frame_image_t2, center_tck_t2, width_tck_t2)
            lab_frame_masks_tr, worm_frame_images_tr, worm_frame_masks_tr = mask_generation(lab_frame_image_tr, center_tck_tr, width_tck_tr)

        #we don't care about the worm frame masks for the t2 and tr worms, so just generate a mask
        """lab_frame_mask_t2 = worm_spline.lab_frame_mask(center_tck_t2, width_tck_t2, lab_frame_image_t2.shape)
                                lab_frame_mask_t2 = lab_frame_mask_t2>0
                                lab_frame_mask_tr = worm_spline.lab_frame_mask(center_tck_tr, width_tck_tr, lab_frame_image_tr.shape)
                                lab_frame_mask_tr = lab_frame_mask_tr>0"""
        
        for lab_frame_mask_t1, worm_frame_image_t1, worm_frame_mask_t1, lab_frame_mask_t2, lab_frame_mask_tr in zip(lab_frame_masks_t1, 
                worm_frame_images_t1, worm_frame_masks_t1, lab_frame_masks_t2, lab_frame_masks_tr):
            #evaluate the measurement of comparison between worm t and worm t+1 and lab_frame_t1 and worm_frame_t1
            compare_value_tt1 = measurement_func(lab_frame_image_t1, lab_frame_mask_t1, lab_frame_image_t2, lab_frame_mask_t2)
            #evaluate the measurement of comparison between unwarped worm t and its warped image
            compare_value_tt = measurement_func(lab_frame_image_t1, lab_frame_mask_t1, worm_frame_image_t1, worm_frame_mask_t1)
            #measurement of comparison between worm t and random worm
            compare_value_ttr = measurement_func(lab_frame_image_t1, lab_frame_mask_t1, lab_frame_image_tr, lab_frame_mask_tr)
            #lab_frame_image_t1[lab_frame_mask_t1] = 1

            measurements.append({'consecutive timepoint measurements': compare_value_tt1, 
                'warp to unwarped' : compare_value_tt, 
                'worm vs random worm': (compare_value_ttr, (random_worm, random_time))})
        timepoint_measurements[tp1] = measurements

    return timepoint_measurements
