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
from tqdm import tqdm
from data_loader import cityscapesDataset
import cv2
import numpy as np
from copy import deepcopy

from model import BASNet

def normPRED(d):
	ma = torch.max(d)
	mi = torch.min(d)

	dn = (d-mi)/(ma-mi)

	return dn

def save_output(save_path, pred):
	os.makedirs(save_dir, exist_ok=True)

	pred = pred.squeeze().cpu().data.numpy()
	im = Image.fromarray(pred*255).convert('RGB')

	save_path = save_path + '.png'
	im.save(save_path)


def postprocess(model_output: np.array) -> np.array:
    """
	from https://github.com/wvangansbeke/Unsupervised-Semantic-Segmentation/tree/main/saliency 
	We postprocess the predicted saliency mask to remove very small segments. 
	If the mask is too small overall, we skip the image.

	Args:
	    model_output: The predicted saliency mask scaled between 0 and 1. 
	                  Shape is (height, width). 
	Return:
            result: The postprocessed saliency mask.
    """
	mask = (model_output > 0.5).astype(np.uint8)
	contours, _ = cv2.findContours(deepcopy(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	# Throw out small segments
	for contour in contours:
	    segment_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
	    segment_mask = cv2.drawContours(segment_mask, [contour], 0, 255, thickness=cv2.FILLED)
	    area = (np.sum(segment_mask) / 255.0) / np.prod(segment_mask.shape)
            if area < 0.01:
		mask[segment_mask == 255] = 0

	# If area of mask is too small, return None
	if np.sum(mask) / np.prod(mask.shape) < 0.01:
	    return None

	return mask

if __name__ == '__main__':
	# --------- 1. get image path and name ---------
	
	#image_dir = './test_data/cityscapes/leftImg8bit_tiny/'
	image_dir = './test_data/gta5/images_tiny/'
	save_dir = './test_data/gta_results_tiny/'
	model_dir = './saved_models/basnet.pth'
		
	# --------- 2. dataloader ---------
	#1. dataload
	transform = transforms.Compose([transforms.ToTensor()])
	dataset = cityscapesDataset(image_path=image_dir, transform=transform)
	dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
	
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
		save_name = dataset.get_img_save_path(data['index'])
		save_output(save_dir, save_name, pred)
	
		del d1,d2,d3,d4,d5,d6,d7,d8
