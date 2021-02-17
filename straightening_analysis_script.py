import numpy
from scipy import stats
import matplotlib.pyplot as plt

from elegant.torch import dataset
from elegant import datamodel
from keypoint_annotation.dataloaders import training_dataloaders

from elegant_scripts import keypoint_reproducibility_helpers
from straightening_analysis import straightening_analysis
from straightening_analysis import mask_generation
from straightening_analysis import measurement_funcs

exp_root1 = '/Volumes/lugia_array/20170919_lin-04_GFP_spe-9/'
exp_root2 = '/Volumes/lugia_array/20190408_lin-4_spe-9_20C_pos-1/'
#exp_root2 = '/mnt/9karray/Mosley_Matt/20190408_lin-4_spe-9_20C_pos-1/'
exp_root3 = '/Volumes/lugia_array/20190813_lin4gfp_spe9_control/20190813_lin4gfp_spe9_control/'
#exp_root3 = '/mnt/scopearray/Mosley_Matt/glp-1/20190813_lin4gfp_spe9_control'
experiments = [datamodel.Experiment(path) for path in (exp_root1, exp_root2, exp_root3)]

#filter experiments to only get adult timepoints
experiments[0].filter(timepoint_filter=(keypoint_reproducibility_helpers.filter_holly_data))
experiments[1].filter(timepoint_filter=keypoint_reproducibility_helpers.filter_adults)
experiments[2].filter(timepoint_filter=keypoint_reproducibility_helpers.filter_adults)

#once we get the ages, need to filter ones with poses (need to do it this way because the calculate adult_age assumes the first timepoint
#is the first adult age)
for experiment in experiments:
    experiment.filter(timepoint_filter=(keypoint_reproducibility_helpers.has_pose, keypoint_reproducibility_helpers.has_keypoints))

#make at timepoint list and save out the timepoint paths
timepoint_list = datamodel.Timepoints.from_experiments(*experiments)

