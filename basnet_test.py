import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import pdb
import numpy as np
from PIL import Image
import glob
import tqdm
from data_loader import cityscapesDataset

from model import BASNet

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(save_dir, save_name, pred):
	os.makedirs(save_dir, exist_ok=True)

	pred = pred.squeeze().cpu().data.numpy()
	im = Image.fromarray(pred*255).convert('RGB')

	save_path = save_dir + '/' + save_name + '.png'
	im.save(save_path)


if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	
	image_dir = './test_data/cityscapes/leftImg8bit_tiny/'
	save_dir = './test_data/test_results_no_crop/'
	model_dir = './saved_models/basnet.pth'
		
	# --------- 2. dataloader ---------
	#1. dataload
	transform = transforms.Compose([transforms.ToTensor()])
	dataset = cityscapesDataset(image_path=image_dir, transform=transform)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
	
	# --------- 3. model define ---------
	print("...load BASNet...")
	net = BASNet(3,1)
	net.load_state_dict(torch.load(model_dir))
	if torch.cuda.is_available():
		net.cuda()
	net.eval()
	
	# --------- 4. inference for each image ---------
	for data in tqdm(dataloader):

		#print('inferencing... ', i)
	
		inputs_test = data['image']
		inputs_test = inputs_test.type(torch.FloatTensor)
		if torch.cuda.is_available():
			inputs_test.cuda()

		inputs_test = Variable(inputs_test)
		d1,d2,d3,d4,d5,d6,d7,d8 = net(inputs_test)
		pred = d1[:,0,:,:]
		
		# normalization
		pred = normPRED(pred)

		# save
		save_dir2, save_name = dataset.get_img_save_path(data['index'])
		save_output_cs(save_dir + save_dir2, save_name, pred)
	
		del d1,d2,d3,d4,d5,d6,d7,d8
