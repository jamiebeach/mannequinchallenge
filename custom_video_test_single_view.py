# How to execute this script
# python custom_video_test_single_view.py "path/to/input/video.mp4" --input=single_view

import torch
from options.train_options import TrainOptions
from loaders import aligned_data_loader
from models import pix2pix_model
import numpy as np
import cv2
import sys
import glob
import os 
import shutil
import sys 

def video_to_images(video_path):
	video_file_name = os.path.split(video_path)[1].split('.')[0]

	# Load the video file from the give path
	cap = cv2.VideoCapture(video_path)

	# Specify where to put the images
	input_path = 'test_data/custom_video_predictions/input_images/'
	filename_path = 'test_data/custom_video_predictions/'+video_file_name+'.txt'
	
	try:
		os.mkdir(input_path)
	except:
		pass

	# Open the images dataset
	images_list = open(filename_path, 'w') 
	counter = 0 

	while(cap.isOpened()):
		# Get the frame from the video
		ret, frame = cap.read()
		if not ret:
			break
		# Prepare the output image filename
		image_name = input_path+video_file_name+'_'+str(counter)+'.jpg'
		# Write the name into the dataset txt file
		images_list.write(image_name+'\n')
		# Save the image in the specified path
		cv2.imwrite(image_name,frame)
		counter += 1 

	images_list.close()
	cap.release()
	cv2.destroyAllWindows()
	print("Number of images: {} ".format(counter))

	# Return the path to the video images folder 
	return filename_path, input_path

def images_to_video(images_path, video_name = 'output_video.avi'):
	img_array = []
	for filename in glob.glob(images_path+'*'):
		print(filename)
		img = cv2.imread(filename)
		height, width, layers = img.shape
		size = (width,height)
		img_array.append(img)
		
	out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
	for i in range(len(img_array)):
		out.write(img_array[i])
	out.release()

# The video transformed into list of images with 'jpg' extension
input_video_path = sys.argv[1]

filename_path, output_path = video_to_images(input_video_path)
BATCH_SIZE = 1

# Load the video dataset, load the images in a list and prepare it 
video_data_loader = aligned_data_loader.CUSTOMDataLoader(filename_path, BATCH_SIZE)
# Load the data (list of images preprocessed to be fed into the model)
video_dataset = video_data_loader.load_data()

print('========================= Video dataset #images = %d =========' %
      len(video_data_loader))

eval_num_threads = 2
opt = TrainOptions().parse()  # set CUDA_VISIBLE_DEVICES before import torch

# Load the prediction model
model = pix2pix_model.Pix2PixModel(opt)

# Set the PyTorch parameters
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
best_epoch = 0
global_step = 0
model.switch_to_eval()

print(
    '=================================  BEGIN VALIDATION ====================================='
)

print('TESTING ON VIDEO')

save_path = 'test_data/custom_video_predictions/output_images/'

try:
	os.mkdir(save_path)
except:
	pass

print('save_path %s' % save_path)

for i, data in enumerate(video_dataset):
	# Get the input image
	input_img = data[0]
	# Run the model using the current input image 
	model.run_custom_img(input_img, save_path, i)

print('Creating the output video...')
images_to_video(save_path)
print('Depth measurement video was created successfully.') 