import os
import random
import numpy as np
import shutil
import copy

train_dir_predict_flag = False
if train_dir_predict_flag == True:
	os.system(cmdstr)
cmdstr = 'python VisualizeResults.py ' + '--data_dir ' + './' + test_dir_str + ' --file_type test.txt' + ' --best_model_loc ' + './results/' + train_dir_name + '/148_epoch_model.pth' + ' --outfile_dir' + ' ./results/' + test_dir_str
os.system(cmdstr)