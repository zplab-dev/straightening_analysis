import pathlib
import freeimage
import numpy
import scipy
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from elegant import process_images

def preprocess_image(timepoint, img_type='bf'):
    position_root = timepoint.position.path
    image_corrected = process_images.flatfield_correct(position_root, timepoint.name, img_type)
    return image_corrected

def measure_timepoint_list(timepoint_list, mask_generation, measurement_funcs, measurement_names, longitudinal_warp=False, img_type='bf'):
    total_tps = len(timepoint_list)
    for i, tp in enumerate(timepoint_list):
        if i%10 == 0:
            print("Completed {}/{} timepoints".format(i, total_tps))
        measurements = measure_timepoint(tp, mask_generation, measurement_funcs, measurement_names, longitudinal_warp, img_type)
        tp.annotations.update(measurements)

def measure_timepoint(timepoint, mask_generation, measurement_funcs, measurement_names, longitudinal_warp=False, img_type='bf'):
    tp1 = timepoint
    tp_idx = list(tp1.position.timepoints.keys()).index(tp1.name)
    tp2 = list(tp1.position.timepoints.values())[tp_idx+1]
    #get random worm
    random_position = random.choice(list(tp1.position.experiment.positions.keys()))
    tpr = random.choice(list(tp1.position.experiment.positions[random_position].timepoints.values()))
    while tpr == tp1 or tpr == tp2:
        random_position = random.choice(list(tp1.position.experiment.positions.keys()))
        tpr = random.choice(list(tp1.position.experiment.positions[random_position].timepoints.values()))

    lab_frame_image_t1 = preprocess_image(tp1, img_type)
    lab_frame_image_t2 = preprocess_image(tp2, img_type)
    lab_frame_image_tr = preprocess_image(tpr, img_type)

    center_tck_t1, width_tck_t1 = tp1.annotations['pose']
    center_tck_t2, width_tck_t2 = tp2.annotations['pose']
    center_tck_tr, width_tck_tr = tpr.annotations['pose']

    #generate the masks for each worm
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
    
    #measure stuff!
    results = {}
    for measure_func, mname in zip(measurement_funcs, measurement_names):
        print("Measuring "+mname)
        measurements = []
        for lab_frame_mask_t1, worm_frame_image_t1, worm_frame_mask_t1, lab_frame_mask_t2, lab_frame_mask_tr in zip(lab_frame_masks_t1, 
                worm_frame_images_t1, worm_frame_masks_t1, lab_frame_masks_t2, lab_frame_masks_tr):
            #evaluate the measurement of comparison between worm t and worm t+1 and lab_frame_t1 and worm_frame_t1
            compare_value_tt1 = measure_func(lab_frame_image_t1, lab_frame_mask_t1, lab_frame_image_t2, lab_frame_mask_t2)
            #evaluate the measurement of comparison between unwarped worm t and its warped image
            compare_value_tt = measure_func(lab_frame_image_t1, lab_frame_mask_t1, worm_frame_image_t1, worm_frame_mask_t1)
            #measurement of comparison between worm t and random worm
            compare_value_ttr = measure_func(lab_frame_image_t1, lab_frame_mask_t1, lab_frame_image_tr, lab_frame_mask_tr)
            #lab_frame_image_t1[lab_frame_mask_t1] = 1
            measurements.append({'consecutive timepoint measurements': compare_value_tt1, 
                'warp to unwarped' : compare_value_tt, 
                'worm vs random worm': (compare_value_ttr, (tpr.position.name, tpr.name))})
        results[mname] = measurements

    return results 

def extract_measurements(timepoint_list, measurement_name):
    """For plotting/general purposes, use this to get the measurements out from the dictionaries that measure_experiment returns
    """
    consec_tp = []
    warp_v_unwarp = []
    tp_v_rand = []
    identifier = []
    measurement_list = {'consecutive timepoint measurements' : [], 'warp to unwarped':[], 'worm vs random worm':[]}

    for timepoint in timepoint_list:
        consec_tp.append(timepoint.annotations[measurement_name]['consecutive timepoint measurements'])
        warp_v_unwarp.append(timepoint.annotations[measurement_name]['warp to unwarped'])
        tp_v_rand.append(timepoint.annotations[measurement_name]['worm vs random worm'][0])
        identifier.append((timepoint))

    #add to the end dictionary
    measurement_list['consecutive timepoint measurements'] = consec_tp
    measurement_list['warp to unwarped'] = warp_v_unwarp
    measurement_list['worm vs random worm'] = tp_v_rand
    measurement_list['Identifier'] = identifier

    return measurement_list

def extract_area_measurements(timepoint_list, measurement_names):
    """Since the area measurements are slightly different than other measurements, we need a different function to 
    extract them and consolidate them from the measurement lists
    """
    extracted_list = {}
    for m in measurement_names:
        consec_tp = []
        warp_v_unwarp = []
        tp_v_rand = []
        identifier = []
        measurement_list = {'consecutive timepoint measurements' : [], 'warp to unwarped':[], 'worm vs random worm':[]}
    for timepoint in timepoint_list:
        consec_area, consec_total = timepoint.annotations['consecutive timepoint measurements']
        consec_tp.append(consec_area/consec_total)

        warp_area, warp_total = timepoint.annotations['warp to unwarped']
        warp_v_unwarp.append(warp_area/warp_total)

        rand_area, rand_total = timepoint.annotations['worm vs random worm'][0]
        tp_v_rand.append(rand_area/rand_total)

        identifier.append(timepoint)

        #add to the end dictionary
        measurement_list['consecutive timepoint measurements'] = consec_tp
        measurement_list['warp to unwarped'] = warp_v_unwarp
        measurement_list['worm vs random worm'] = tp_v_rand
        measurement_list['Identifier'] = identifier
    extracted_list[m] = measurement_list

    return extracted_list

def extract_slice_measurements(timepoint_list, measurement_names):
    
    extracted_list = {}
    for m in measurement_names:
        measurement_list = {'consecutive timepoint measurements' : [], 'warp to unwarped':[], 'worm vs random worm':[]}
        consec_tp = []
        warp_v_unwarp = []
        tp_v_rand = []
        identifier = []
        for timepoint in timepoint_list:
            consec_measurements = 0
            warp_measurements = 0
            rand_measurements = 0

            for slice_m in timepoint.annotations[m]:
                consec_measurements += slice_m['consecutive timepoint measurements']
                warp_measurements += slice_m['warp to unwarped']
                rand_measurements += slice_m['worm vs random worm'][0]

            consec_tp.append(consec_measurements)
            warp_v_unwarp.append(warp_measurements)
            tp_v_rand.append(rand_measurements)
            identifier.append((timepoint))

        #add to the end dictionary
        measurement_list['Identifier'] = identifier
        measurement_list['consecutive timepoint measurements'] = consec_tp
        measurement_list['warp to unwarped'] = warp_v_unwarp
        measurement_list['worm vs random worm'] = tp_v_rand

    extracted_list[m] = measurement_list
    return extracted_list

def extract_slice_area_measurements(timepoint_list, measurement_names):
    """Used to extract measurements from gfp_area from slices
    """
    
    extracted_list = {}
    
    for m in measurement_names:
        consec_tp = []
        warp_v_unwarp = []
        tp_v_rand = []
        identifier = []
        measurement_list = {'consecutive timepoint measurements' : [], 'warp to unwarped':[], 'worm vs random worm':[]}

        for timepoint in timepoint_list:
            consec_area = 0
            consec_total_area = 0
            
            warp_area = 0
            warp_total_area = 0

            rand_area = 0
            rand_total_area = 0
            for slice_m in timepoint.annotations[m]:
                c, ca = slice_m['consecutive timepoint measurements']
                consec_area += c
                consec_total_area += ca
                w, wa = slice_m['warp to unwarped']
                warp_area += w
                warp_total_area += wa
                r, ra = slice_m['worm vs random worm'][0]
                rand_area += r
                rand_total_area += ra

            identifier.append(timepoint)
            consec_tp.append(consec_area.astype(numpy.float64)/consec_total_area.astype(numpy.float64))
            warp_v_unwarp.append(warp_area.astype(numpy.float64)/warp_total_area.astype(numpy.float64))
            tp_v_rand.append(rand_area.astype(numpy.float64)/rand_total_area.astype(numpy.float64))

        #add to the end dictionary
        measurement_list['Identifier'] = identifier
        measurement_list['consecutive timepoint measurements'] = consec_tp
        measurement_list['warp to unwarped'] = warp_v_unwarp
        measurement_list['worm vs random worm'] = tp_v_rand
    extracted_list[m] = measurement_list

    return extracted_list

def summary_stats(measurement_list):
    """Generate some summary stats (mean, std deviation)
    """
    stats_list={}
    for mname, m_list in measurement_list.items():
        stats = {}
        for key, measurements in m_list.items():
            if key is 'Identifier':
                break
            sum_stats = {}
            sum_stats['mean'] = numpy.mean(measurements)
            sum_stats['std'] = numpy.std(measurements)
            sum_stats['min'] = numpy.min(measurements)
            sum_stats['max'] = numpy.max(measurements)
            sum_stats['variance'] = numpy.var(measurements)

            stats[key] = sum_stats
        stats_list[mname] = stats
    return stats_list

def plot_distributions(measurement_list, title, plt_figure):
    """ Using Matplotlib plot the things
    """
    xs, ys, kd_estimator = kde.kd_distribution(measurement_list['warp to unwarped'])
    plt_figure.plot(xs, ys, color='r', label='warp to unwarped')
    xs, ys, kd_estimator = kde.kd_distribution(measurement_list['consecutive timepoint measurements'])
    plt_figure.plot(xs, ys, color='b', label='consecutive timepoint measurements')
    xs, ys, kd_estimator = kde.kd_distribution(measurement_list['worm vs random worm'])
    plt_figure.plot(xs, ys, color='g', label='worm vs random worm')

    plt_figure.set_title(title)
    plt_figure.legend()


def plot(measurements, save_dir, features, days="all"):
    """Function to plot things in case you want to only plot
    a few features

    Parameters:
        measurements: tuple of list of measurements to plot (x,y)
            In general, this is (lifespan, gfp_measurement)
        save_dir: place to save the files
        features: names of the features you are trying to plot,
            inputted in the same way as the measurments (x,y)
    
    Returns:
    """
    save_dir = pathlib.Path(save_dir)
    save_path = save_dir / (features[0]+" vs "+features[1]+" "+days+" dph.png")

    #get the regression line stuff
    pearson,spearman,yp=run_stats(measurements[0],measurements[1])

    plt.scatter(measurements[0], measurements[1])
    plt.plot(measurements[0],yp, c='gray')
    title = features[0]+" vs "+features[1]+" at "+days+" dph"
    plt.style.use('seaborn-white')
    plt.xlabel((features[0]+" (dph)"), fontdict={'size':20,'family':'calibri'})
    plt.ylabel(("Mean "+features[1]), fontdict={'size':20,'family':'calibri'})
    plt.title(title, y=1.05,fontdict={'size':26,'weight':'bold','family':'calibri'})


    if pearson[1]<.00001:
        p="p<.00001"
    else:
        p="p=" + ''+ (str)(round(pearson[1],3))
    if spearman[1]<.00001:
        spearman_p="p<.00001"
    else:
        spearman_p="p=" + '' + (str)(round(spearman[1],3))
                
    ftext='$r^{2}$ = '+(str)(round(pearson[0]**2,3))+" "+p
    gtext=r'$\rho$'+" = "+(str)(round(spearman[0],3))+" "+spearman_p
    
    plt.figtext(.15,.85,ftext,fontsize=20,ha='left')
    plt.figtext(.15,.8,gtext,fontsize=20,ha='left')
 
    plt.gcf()
    plt.savefig(save_path.as_posix())
    plt.show(block=False)
    time.sleep(1)
    plt.close()
    plt.gcf().clf


def plot_violin(measurement_list, title, save_dir, measurement_name, ylabel='Fraction pixel changed'):
    """Createa violin plot of the data from the measurement list.
    Note: the extract functions give this list
    """
    fig, ax = plt.subplots()
    positions = numpy.array([1,2,3])

    plot_list = []
    colors = ['blue','orange']
    
    ax.violinplot([measurement_list['warp to unwarped'], measurement_list['consecutive timepoint measurements'], 
            measurement_list['worm vs random worm']], positions, showmeans=True, showextrema=False)
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['Warped to Unwarped', 'Consecutive Timepoints', 'Random Worm'])
    ax.set_title(title)

    ax.set_ylabel(ylabel)

    save_dir = pathlib.Path(save_dir)
    save_file = save_dir/str(measurement_name+'.png')
    plt.savefig(str(save_file))
    save_file = save_dir/str(measurement_name+'.svg')
    plt.savefig(str(save_file))

    plt.gcf().clf
    plt.close()

def plot_boxplot(measurement_list, title, save_file):
    """Createa violin plot of the data from the measurement list.
    Note: the extract functions give this list
    """
    fig, ax = plt.subplots()
    ax.boxplot([measurement_list['warp to unwarped'], measurement_list['consecutive timepoint measurements'], measurement_list['worm vs random worm']])
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['Warped to Unwarped', 'Consecutive Timepoints', 'Random Timepoints'])
    ax.set_title(title)
    plt.savefig(save_file)

    plt.gcf().clf
    plt.close()


def run_stats(x_list,y_list):
    """Get the pearson, spearman, and polyfit coorelations from
    the data.
    """
    pearson=numpy.asarray(scipy.stats.pearsonr(x_list, y_list))
    spearman=numpy.asarray(scipy.stats.spearmanr(x_list, y_list))
    (m,b) = numpy.polyfit(x_list, y_list, 1)
    yp=numpy.polyval([m,b], x_list)
    
    return (pearson,spearman, yp)

