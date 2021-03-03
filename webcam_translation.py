# Copyright 2021 by YoungWoon Cho, Danny Hong
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture

import cv2
import time
from PIL import Image
import torchvision.transforms as transforms
from models import Generator
import numpy as np
import torch

IMG_SIZE = 256

device = torch.device("cpu")
model = Generator().to(device)
model.load_state_dict(torch.load('weights/fruit2rotten/G_A2B.pth', map_location=torch.device('cpu')))
model.eval()

# Load image
def translate_webcam(image):
  transform = transforms.Compose([transforms.Resize(IMG_SIZE),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
  image = Image.fromarray(image.astype('uint8'), 'RGB')
  image = transform(image).unsqueeze(0)
  image = image.to(device)
  translated_image = model(image)
  return translated_image

####################
#WEBCAM TRANSLATION#
####################
# define a video capture object
vid = cv2.VideoCapture('http://ahzin:ahzin@192.168.0.4:8081/') 
while(True):
	print('Translating...')
	start_time = time.time() # Reset the time
	ret, frame = vid.read()

	translated_image = translate_webcam(frame)
	translated_image = translated_image[0].detach().numpy().transpose(1, 2, 0)
	#full_image = cv2.resize(translated_image, (1920, 1080))
	cv2.imshow('translated_image', translated_image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
vid.release() 
cv2.destroyAllWindows()