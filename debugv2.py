# import numpy as np
# import pydensecrf.densecrf as dcrf
#
# d = dcrf.DenseCRF2D(640, 480, 5)  # width, height, nlabels
# for i in range(3):
# 	print(i)
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class UNet(nn.Module):
	def __init__(self, in_channels=1, n_classes=2, depth=5, wf=6, padding=False,
				 batch_norm=False, up_mode='upconv'):
		"""
		Implementation of
		U-Net: Convolutional Networks for Biomedical Image Segmentation
		(Ronneberger et al., 2015)
		https://arxiv.org/abs/1505.04597
		Using the default arguments will yield the exact version used
		in the original paper
		Args:
			in_channels (int): number of input channels
			n_classes (int): number of output channels
			depth (int): depth of the network
			wf (int): number of filters in the first layer is 2**wf
			padding (bool): if True, apply padding such that the input shape
							is the same as the output.
							This may introduce artifacts
			batch_norm (bool): Use BatchNorm after layers with an
							   activation function
			up_mode (str): one of 'upconv' or 'upsample'.
						   'upconv' will use transposed convolutions for
						   learned upsampling.
						   'upsample' will use bilinear upsampling.
		"""
		super(UNet, self).__init__()
		assert up_mode in ('upconv', 'upsample')
		self.padding = padding
		self.depth = depth
		prev_channels = in_channels
		self.down_path = nn.ModuleList()
		for i in range(depth):
			self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i),
												padding, batch_norm))
			prev_channels = 2 ** (wf + i)

		self.up_path = nn.ModuleList()
		for i in reversed(range(depth - 1)):
			self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode,
											padding, batch_norm))
			prev_channels = 2 ** (wf + i)

		self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

	def forward(self, x):
		blocks = []
		for i, down in enumerate(self.down_path):
			x = down(x)
			if i != len(self.down_path) - 1:
				blocks.append(x)
				x = F.avg_pool2d(x, 2)

		for i, up in enumerate(self.up_path):
			x = up(x, blocks[-i - 1])

		return self.last(x)


class UNetConvBlock(nn.Module):
	def __init__(self, in_size, out_size, padding, batch_norm):
		super(UNetConvBlock, self).__init__()
		block = []

		block.append(nn.Conv2d(in_size, out_size, kernel_size=3,
							   padding=int(padding)))
		block.append(nn.ReLU())
		if batch_norm:
			block.append(nn.BatchNorm2d(out_size))

		block.append(nn.Conv2d(out_size, out_size, kernel_size=3,
							   padding=int(padding)))
		block.append(nn.ReLU())
		if batch_norm:
			block.append(nn.BatchNorm2d(out_size))

		self.block = nn.Sequential(*block)

	def forward(self, x):
		out = self.block(x)
		return out


class UNetUpBlock(nn.Module):
	def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
		super(UNetUpBlock, self).__init__()
		if up_mode == 'upconv':
			self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
										 stride=2)
		elif up_mode == 'upsample':
			self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
									nn.Conv2d(in_size, out_size, kernel_size=1))

		self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

	def center_crop(self, layer, target_size):
		_, _, layer_height, layer_width = layer.size()
		diff_y = (layer_height - target_size[0]) // 2
		diff_x = (layer_width - target_size[1]) // 2
		return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

	def forward(self, x, bridge):
		print(bridge.size())
		up = self.up(x)
		crop1 = self.center_crop(bridge, up.shape[2:])
		out = torch.cat([up, crop1], 1)
		out = self.conv_block(out)

		return out


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = UNet(n_classes=2, padding=True, up_mode='upsample').to(device)
# optim = torch.optim.Adam(model.parameters())
# fake_im_num = 1
# numpy_fake_image_3d = np.random.rand(fake_im_num, 1, 120, 120)
# tensor_fake_image_3d = torch.FloatTensor(numpy_fake_image_3d)
# torch_fake_image_3d = Variable(tensor_fake_image_3d).cuda()
# output_3d = model(torch_fake_image_3d)

import os
import random
import numpy as np
import shutil
import copy
# Obtain test images ID
# Enter the root path
cwd = os.getcwd()
# print('The name of current dictionary is', cwd)
brats_test_exp1_data_path = cwd + '/data/original_brats17/HGG'
test37_file_names = os.listdir(brats_test_exp1_data_path)
# print('The number of files used for training is ', len(test37_file_names))

# Obtain train images ID
brats_train_exp1_data_file_path = '/home/donghao/Desktop/donghao/isbi2019/code/brats17/config17/train_names_106.txt'
# Open the file with read only permit
f = open(brats_train_exp1_data_file_path)
# use readline() to read the first line
line = f.readline()
# use the read line to read further.
# If the file is not empty keep reading one line
# at a time, till the file is empty
train106_im_name_list = []
while line:
	# in python 3 print is a builtin function, so
	line = line.strip('\n')
	train106_im_name_list.append(line)
	# print(os.path.basename(line))

	# use realine() to read next line
	line = f.readline()
f.close()
#print(test37_file_names)
#print(train106_im_name_list)

full_image_names_list_path = '/home/donghao/Desktop/donghao/brain_segmentation/brain_data_full/HGG'
# Open the file with read only permit
full_image_names_list = os.listdir(full_image_names_list_path)
train106_image_names_list_update = []
print('The legnth of all images before ', len(full_image_names_list))

for imname in train106_im_name_list:
	#print(os.path.basename(imname))
	#print('imname is ', imname)
	#print(train106_image_names_list_update)
	train106_image_names_list_update.append(os.path.basename(imname))
	full_image_names_list.remove(os.path.basename(imname))
print('The length of all images after removing training set is :', len(full_image_names_list))

for imname in test37_file_names:
	full_image_names_list.remove(imname)
print('the length of all images after removing testing set is :', len(full_image_names_list))
# for imname in test37_file_names:
# 	full_image_names_list.remove(imname)
# print('The length of full_image_names_list is ', len(full_image_names_list))

# The new training list of 58 images
np.random.seed(6)
sz_train = 58
sz_train_copy = copy.copy(sz_train)
sz_rm = 210-106-37
sz_test_copy = sz_rm - sz_train
new_trainim_num = np.random.randint(1, sz_rm, size=sz_train)
new_trainim_num = np.unique(new_trainim_num)
while len(new_trainim_num) < sz_train_copy:
	np.random.seed(6)
	sz_train=sz_train+1
	new_trainim_num = np.random.randint(1, sz_rm, size=sz_train)
	new_trainim_num = np.unique(new_trainim_num)
new_trainim_num.astype(int)
#print('the length of new_trainim_num is ', len(new_trainim_num))
#print(new_trainim_num)
new_train_list = []
#print(new_train_list)


new_test_list = full_image_names_list.copy()
#print('x',full_image_names_list[63])
for new_train_list_item in new_trainim_num:
	#print('new_train_list_item: ', new_train_list_item)
	new_train_list.append(full_image_names_list[new_train_list_item])
	new_test_list.remove(full_image_names_list[new_train_list_item])

print('the length of training image list is ', len(new_train_list))
print(new_train_list)
print('the length of testing image list is ', len(new_test_list))
print(new_test_list)

# create training folder
train_dir_str = 'data/train_'+str(sz_train_copy)+'_exp2'
print(train_dir_str)
if not os.path.exists(train_dir_str):
	os.mkdir(train_dir_str)

# create testing folder
test_dir_str = 'data/test_'+str(sz_rm-sz_train_copy)+'_exp2'
if not os.path.exists(test_dir_str):
	os.mkdir(test_dir_str)

datafullhgg_path = '/home/donghao/Desktop/donghao/brain_segmentation/brain_data_full/HGG'
# # copy training list to the folder
for train_im in new_train_list:
	srcpath = datafullhgg_path + os.sep + train_im
	destpath = train_dir_str + '/HGG' + os.sep + train_im
	if not os.path.exists(destpath):
		shutil.copytree(src=srcpath, dst=destpath)
#
# # copy testing list
for test_im in new_test_list:
	srcpath = datafullhgg_path + os.sep + test_im
	destpath = test_dir_str + '/HGG' + os.sep + test_im
	if not os.path.exists(destpath):
		shutil.copytree(src=srcpath, dst=destpath)
# the task to do
cmdstr = 'python preprocessing.py --data_dir ' + str(train_dir_str)


code_data_path = '/home/donghao/Desktop/donghao/isbi2019/code/3D-ESPNet'
if not os.path.exists(code_data_path+'/'+train_dir_str+'_preprocess'):
	print(cmdstr)
	os.system(cmdstr)
# # copy preprocessing file into the training folder

srcpath = code_data_path + '/gen.py'
dstpath = train_dir_str + '_preprocess/gen.py'
shutil.copyfile(src=srcpath, dst=dstpath)
#
# # run gen.py inside the corresponding folder
gen_file_path = code_data_path + os.sep + train_dir_str + '_preprocess'
os.chdir(gen_file_path)
cwd = os.getcwd()

os.system("python gen.py --run_type train.txt")
# # the follwing line is to train the task
train_flag = False
if train_flag == True:
	print('The current path is ', cwd)
	os.chdir(code_data_path)
	cmdstr = 'CUDA_VISIBLE_DEVICES=1 python main.py --data_dir "./data/train_58_exp2_preprocess" --data_dir_val "./data/train_58_exp2_preprocess" --max_epochs 150 --savedir ./results/train_58_exp2 --batch_size 3 --lr 5e-3'
	os.system(cmdstr)
#CUDA_VISIBLE_DEVICES=1 python main.py --data_dir "./data/train_58_exp2_preprocess" --data_dir_val "./data/train_58_exp2_preprocess" --max_epochs 150 --savedir ./results/train_58_exp2 --batch_size 3
# copy train.txt from train data directory to train preprocess
os.chdir(code_data_path)
dstpath = code_data_path + os.sep + train_dir_str + '/train.txt'
srcpath = code_data_path + os.sep + train_dir_str + '_preprocess/train.txt'
shutil.copyfile(src=srcpath, dst=dstpath)

# generate the testing dataset txt file
os.chdir(code_data_path)
srcpath = code_data_path + '/gen_official.py'
dstpath = test_dir_str + '/gen_official.py'
shutil.copyfile(src=srcpath, dst=dstpath)
os.chdir(code_data_path + os.sep + test_dir_str)
os.system("python gen_official.py")
# # enter the root path
os.chdir('/home/donghao/Desktop/donghao/isbi2019/code/3D-ESPNet')
# # os.system('python VisualizeResults.py --data_dir ./data/test_29_exp1/ --test_file test.txt --best_model_loc ./results/train_38_exp1/36_epoch_model.pth')
train_dir_name = 'train_'+ str(sz_train_copy) + '_exp2'
test_dir_name = 'test_'+ str(sz_test_copy) + '_exp2'
cmdstr = 'python VisualizeResults.py ' + '--data_dir ' + './' + train_dir_str + ' --file_type train.txt' + ' --best_model_loc ' + './results/' + train_dir_name + '/148_epoch_model.pth' + ' --outfile_dir' + ' ./results/' + 'train_'+ str(sz_train_copy) + '_exp2'

train_dir_predict_flag = False
if train_dir_predict_flag == True:
	print(cmdstr)
	os.system(cmdstr)

test_dir_predict_flag = True
# cmdstr = 'CUDA_VISIBLE_DEVICES=0 python VisualizeResults.py ' + '--data_dir ' + './' + test_dir_str + ' --file_type test.txt' + ' --best_model_loc ' + './results/' + train_dir_name + '/200_epoch_model.pth' + ' --outfile_dir' + ' ./results/' + test_dir_name + 'epoch200'
# cmdstr = 'CUDA_VISIBLE_DEVICES=0 python VisualizeResults.py ' + '--data_dir ' + './' + test_dir_str + ' --file_type test.txt' + ' --best_model_loc ' + './results/' + train_dir_name + '_v1' + '/150_epoch_model.pth' + ' --outfile_dir' + ' ./results/' + test_dir_name + '_epoch150'
# cmdstr = 'CUDA_VISIBLE_DEVICES=0 python VisualizeResults.py ' + '--data_dir ' + './' + test_dir_str + ' --file_type test.txt' + ' --best_model_loc ' + './results/' + train_dir_name + '_v1' + '/100_epoch_model.pth' + ' --outfile_dir' + ' ./results/' + test_dir_name + '_epoch100'
# cmdstr = 'CUDA_VISIBLE_DEVICES=0 python VisualizeResults.py ' + '--data_dir ' + './' + test_dir_str + ' --file_type test.txt' + ' --best_model_loc ' + './results/' + train_dir_name + '_v1' + '/50_epoch_model.pth' + ' --outfile_dir' + ' ./results/' + test_dir_name + '_epoch50'
cmdstr = 'CUDA_VISIBLE_DEVICES=0 python VisualizeResults.py ' + '--data_dir ' + './' + test_dir_str + ' --file_type test.txt' + ' --best_model_loc ' + './results/' + train_dir_name + '_v1' + '/best_model.pth' + ' --outfile_dir' + ' ./results/' + test_dir_name + '_best_v1'
if test_dir_predict_flag == True:
	os.system(cmdstr)
# os.system('python VisualizeResults.py --data_dir ./data/train_38_exp1/ --test_file train.txt --best_model_loc ./results/train_38_exp1/36_epoch_model.pth --outfile_dir ./results/train_38_exp1')