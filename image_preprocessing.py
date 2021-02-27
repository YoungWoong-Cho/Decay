from PIL import Image
import glob
import numpy as np
import os

class Preprocessing():
	def __init__(self, dataset):
		self.dataset_name = dataset
		self.dataset_path = ['trainA', 'trainB', 'testA', 'testB']

		self.image_list = []

		print(self.data_name)

		for path in self.dataset_path:
			path = './input/' + self.dataset_name + '/' + path
			for filename in glob.glob(path+'/*.*'):
				im=Image.open(filename)
				im = np.asarray(im)
				print(im.shape)
				# if len(im.shape) != 3:
				# 	print('Dimension not 3: ' + filename)
				# 	print('Removing '+ filename + '...')
				# 	os.remove(filename)
				# if im.shape[-1] != 3:
				# 	print('Channel not 3: ' + filename)
				# 	if len(im.shape) == 4:
				# 		print('RGBA detected: reducing the dimensionality...')
				# 		im=im[:3]
				# 	else:
				# 		print('Image either grayscale or not compatible.')
				# 		print('Removing '+ filename + '...')
				# 		os.remove(filename)