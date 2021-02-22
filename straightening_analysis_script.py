import os
import platform
import numpy
import sys
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt

from elegant.torch import dataset
from elegant import datamodel
from keypoint_annotation.dataloaders import training_dataloaders

import straightening_analysis_utils
import mask_generation
import measurement_funcs

def has_pose(timepoint):
    pose = timepoint.annotations.get('pose', None)
    # make sure pose is not None, and center/width tcks are both not None
    return pose is not None and pose[0] is not None and pose[1] is not None
    

def has_keypoints(timepoint):
    keypoints = timepoint.annotations.get('keypoints', None)
    return keypoints is not None and not None in keypoints.values() and not False in [x in keypoints.keys() for x in ['anterior bulb', 'posterior bulb', 'vulva', 'tail']]

def run_straightening_analysis(os_type, save_dir):
    print(os_type)
    if os_type == 'Darwin':
        exp_root1 = '/Volumes/lugia_array/20170919_lin-04_GFP_spe-9/'
        exp_root2 = '/Volumes/lugia_array/20190408_lin-4_spe-9_20C_pos-1/'
        #exp_root2 = '/mnt/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1/'
        exp_root3 = '/Volumes/lugia_array/20190813_lin4gfp_spe9_control/20190813_lin4gfp_spe9_control/'
        #exp_root3 = '/mnt/scopearray/Mosley_Matt/glp-1/20190813_lin4gfp_spe9_control'
    elif os_type == 'Linux':
        exp_root1 = '/mnt/lugia_array/20170919_lin-04_GFP_spe-9/'
        exp_root2 = '/mnt/lugia_array/20190408_lin-4_spe-9_20C_pos-1/'
        #exp_root2 = '/mnt/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1/'
        exp_root3 = '/mnt/lugia_array/20190813_lin4gfp_spe9_control/20190813_lin4gfp_spe9_control/'
    
    experiments = [datamodel.Experiment(path) for path in (exp_root1, exp_root2, exp_root3)]

    for experiment in experiments:
        experiment.filter(timepoint_filter=(has_pose, has_keypoints))

    #make at timepoint list and save out the timepoint paths
    timepoint_list = datamodel.Timepoints.from_experiments(*experiments)

    #measure all the things
    measures = [measurement_funcs.measure_emd, measurement_funcs.measure_integrated_gfp, measurement_funcs.measure_area]
    masks = [mask_generation.generate_checkerboard_slice_masks]
    channels = ['bf', 'gfp']
    mnames = ['emd','pixel intensity', 'area']

    straightening_analysis_utils.measure_timepoint_list(timepoint_list, mask_generation.generate_checkerboard_slice_masks, measures, mnames)

    measurement_list = straightening_analysis_utils.extract_slice_measurements(timepoint_list, [mnames[0]])
    area = straightening_analysis_utils.extract_slice_area_measurements(timepoint_list, mnames[1:])
    measurement_list.update(area) #update the measurement list to include everything we need

    summary_stats = straightening_analysis_utils.summary_stats(measurement_list)

    #save out summary stats
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    log_filename = os.path.join(save_dir, 'summary_stats.log')

    fn = open(log_filename, 'a')
    time = datetime.now()
    fn.write('---------------- Straightening Analysis run on {} ---------------------\n'.format(time))
    for measurement, sum_stat in summary_stats.items():
        fn.write('{}\n'.format(measurement))
        for stat, val in sum_stat.items():
            fn.write('\t {}\n'.format(stat))
            for k, v in val.items():
                fn.write('\t\t {}:{}\n'.format(k,v))
        fn.write('\n')

    fn.close()

    #save the figures
    for measurement, m_list in measurement_list.items():
        title = 'Distribution of {} in straightening analyses'.format(measurement)
        if measurement == 'emd':
            straightening_analysis_utils.plot_violin(m_list, title, save_dir, measurement, ylabel='EMD Values')
        else:
            straightening_analysis_utils.plot_violin(m_list, title, save_dir, measurement)

if __name__ == "__main__":
    try:
        save_dir = str(sys.argv[1])
        os_type = platform.system()
        run_straightening_analysis(os_type, save_dir)
    except IndexError:
        print("No save directory found")
        sys.exit(1)
    