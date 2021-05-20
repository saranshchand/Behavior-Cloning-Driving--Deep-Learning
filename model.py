import csv
import cv2
import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

line_array = []
with open('./data_simulator_good_driving/driving_log.csv') as csvf:
	reader = csv.reader(csvf)
	for line in reader:
		line_array.append(line)
imgs = []
angles = []
for line in line_array:
	for num in range(3):
		path = line[num]
		split = path.split('/')
		file_name = split[-1]
		local_path = './data_simulator_good_driving/IMG/' + file_name
		image = cv2.imread(local_path)
		imgs.append(image)
	angle = float(line[3])
	angles.append(angle)
	correction = 0.1
	angles.append(angle+correction)
	angles.append(angle-correction)

line_array = []
with open('./data_simulator_counterclockwise/driving_log.csv') as csvf:
	reader = csv.reader(csvf)
	for line in reader:
		line_array.append(line)		

for line in line_array:
	for num in range(3):
		path = line[num]
		split = path.split('/')
		file_name = split[-1]
		local_path = './data_simulator_counterclockwise/IMG/' + file_name
		image = cv2.imread(local_path)
		imgs.append(image)
	angle = float(line[3])
	angles.append(angle)
	correction = 0.1
	angles.append(angle+correction)
	angles.append(angle-correction)

line_array = []
with open('./data_simulation_correction/driving_log.csv') as csvf:
	reader = csv.reader(csvf)
	for line in reader:
		line_array.append(line)		

for line in line_array:
	for num in range(3):
		path = line[num]
		split = path.split('/')
		file_name = split[-1]
		local_path = './data_simulation_correction/IMG/' + file_name
		image = cv2.imread(local_path)
		imgs.append(image)
	angle = float(line[3])
	angles.append(angle)
	correction = 0.1
	angles.append(angle+correction)
	angles.append(angle-correction)

# imgs and angles array match dimensions so data loading ends here
# Data augmentation 

imgs_aug = [] #augmented images array
angles_aug = [] #augmented angles array
for img, angle in zip(imgs, angles):
	imgs_aug.append(img)
	angles_aug.append(angle)
	img_f = cv2.flip(image, 1)
	angle_f = angle*-1.0
	imgs_aug.append(img_f)
	angles_aug.append(angle_f)

X_train = np.array(imgs_aug)
y_train = np.array(angles_aug)

print(X_train.shape)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5, subsample= (2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))  

model.compile(optimizer='adam', loss = 'mse')
model.fit(X_train, y_train, validation_split=0.2, epochs=3,shuffle=True)

model.save('model.h5')

