import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D,Dropout,Flatten,Dense
import cv2
import pandas as pd
import random
import os
import sys


import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

#Main Data

main_number_bins = 25
main_number_samples = 5000
main_batches_train = 200
main_batches_valid = 200
main_steps_train = 200
main_steps_valid = 200
main_number_epochs = 16
main_probability = 0.5

#from tensorflow.python.client import device_lib

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))# Checking if the tf can detect my GPU or not using cudnn

# ______________________________ Title ______________________________


# ______________________________ Reading Data ______________________________

datadir = "Main Data" # The Folder That contains the data
columns = ['center','left','right','steering','throttle','reverse','speed'] # The Columns

main_data = pd.read_csv(os.path.join(datadir,"driving_log.csv"), names = columns) # Add the csv file to a panda variable
data = main_data # Making A copy of testing Purposes
pd.set_option('display.max_colwidth',None) # For Testing Purposes

data.head() # Seeing the first rows
# print(data.head(10))

# ______________________________ Getting Only Image Titles ______________________________
def path_leaf(path): #  https://www.oreilly.com/library/view/python-standard-library/0596000960/ch13s03.html#:~:text=The%20ntpath%20module%20(see%20Example,Windows%20paths%20on%20other%20platforms.
  head,tail = ntpath.split(path) # getting the name only
  return tail

data['center'] = data['center'].apply(path_leaf) # applying the change
data['left'] = data['left'].apply(path_leaf) # applying the change
data['right'] = data['right'].apply(path_leaf) # applying the change

data.head() # Seeing the first rows

# ______________________________ Balancing Data Test Session 1 (Failed after 5 hours :D) ______________________________
# **********TESTING HISTOGRAMS************


# list2d = []
# for i in range(len(bins)-1):
#   newlist = []

  # print(i)
  # print("bins i : ",bins[i])
  # print("bins i + 1 : ", bins[i+1])

  # for j in range(len(myOwnList)):
  #   # print(myOwnList[j])
  #   # print(f"is { myOwnList[j]} bigger or equal {bins[i]} and { myOwnList[j]} smaller than {bins[i+1]}")
  #   if myOwnList[j] >= bins[i] and myOwnList[j] < bins[i+1]:
  #     newlist.append( myOwnList[j])
  # # print(newlist)
  # list2d.append(newlist)

# plt.hist(list2d, bins= np.linspace(-1,1,11),width=0.05)

# labels, counts = np.unique(myOwnList, return_counts=True)
# plt.bar(labels, counts, align='center',width=0.05)
# plt.gca().set_xticks(bins)
# plt.show()

# plt.hist(myOwnList, bins=bins, edgecolor="k")
# plt.xticks(bins)

# plt.show()
# myOwnList = data['steering']

# bins = np.linspace(-1,1,26)
# print(bins)

# plt.hist(myOwnList, bins= bins,width=0.05)

# ______________________________ Balancing Data Test Session 2 (Worked after 2 hours) ______________________________
# Finally Normalized without changing and centering the data

# num_bins = 25 #this must be an odd number
# hist, bins = np.histogram(data['steering'],num_bins) #I am finding the range of steering
# bins_array = np.linspace(-1.0,1.0,num_bins)
# # Debug: print(bins_array)
# plt.bar(bins_array,hist,width=0.05)
#
# samples_per_bin = 400 # For Balacing we set a limit of 200
#
# plt.plot((np.min(data['steering']),np.max(data['steering'])),(samples_per_bin, samples_per_bin))
# plt.show()



# ______________________________ Balancing Data Test Session 3 (Found on the internet (Not my best Sol)) ______________________________
#Other Way That I don't like

num_bins = main_number_bins #this must be an odd number
hist, bins = np.histogram(data['steering'],num_bins) #I am finding the range of steering
samples_per_bin = main_number_samples

# print(len(bins))
# Debug: print(bins)
# print(bins)
# Debug: print(bins[:-1])
# Debug: print(bins[1:])
# center = (bins[0:] + bins[0:])

center = (bins[:-1] + bins[1:])

# Debug: print(center)

center = center *0.5

# Debug: print(center)
# Debug: print(type(center))
# Debug: print(bins[1:])
# Debug: print(type(np.array([1,2,3])))
plt.bar(center,hist,width=0.05)
plt.plot((np.min(data['steering']),np.max(data['steering'])),(samples_per_bin, samples_per_bin))


# ______________________________ Balancing Data Removing the additional Samples Algorithm  ______________________________


remove_list = []

# print(num_bins)
# print(bins)
# I am updating all num_bins to bins_array
for j in range(num_bins):
  temp_list = [] # Making a temp list for adding each selected data between each bins ( it is like I am classifying them)
  # counter = 0
  # print(bins_array[j])
  # Debug: print("length of bins : ",len(bins))
  # Debug: print("j : ",j)
  # Debug: print("bins at j: ",bins[j])
  # Debug: print("bins at J + 1 : ", bins[j+1])
  for i in range(len(data['steering'])):
    # Debug: print("start___")
    # Debug: print("bin: ",bins[j])
    # Debug: print("data: ",data['steering'][i])
    # Debug: print("bin: ",bins[j+1])
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]: # It was <= but that may get dublicate data
      temp_list.append(i) # I add the numbers between each BINS here
    # Debug: print("list : ", temp_list)
    # Debug: counter = counter + 1
    # Debug: print("counter: ", counter)
    # Debug: print("end step_______________________________________")
    # Debug: if(counter == 50):
    # Debug:   break
  temp_list = shuffle(temp_list) # I am shuffle my data
  temp_list = temp_list[samples_per_bin:] # I am leaving 200 and selecting the rest so that I will remove them
  remove_list.extend(temp_list) # I am adding them to the remove list

  # Debug: print("___________bin: ",bins[j])
  # Debug: print("bin+1: ",bins[j+1])
  # Debug: print("length: ",len(temp_list))
  # Debug: print(data['steering'][temp_list])

print('data before: ',len(data['steering']))
print('removed:',len(remove_list))
data.drop(data.index[remove_list],inplace= True) # I am dropping all the data that is in remove list
print('remaining: ',len(data))


hist , _ = np.histogram(data['steering'],(num_bins))
plt.bar(center,hist,width=0.05)
plt.plot((np.min(data['steering']),np.max(data['steering'])),(samples_per_bin, samples_per_bin))
# plt.show()


# ______________________________ Loading Data Into Variables From The Pandas Variable ______________________________


def load_img_steering(datadir,df): #in here I am getting my data and creating variables for the important parts like getting the images and the steering
  image_path = [] # The Centered one only
  steering = []  # The steering data between -1 and 1

  # Debug: print(df.iloc[1]) # 0 center, 1 left, 2 right, 3 steering, 4 throttle, 5 reverse ,6 speed

  for i in range(len(data)):
    indexed_data = data.iloc[i] # Iloc makes me select each column as an index
    center, left, right,reverse = indexed_data[0], indexed_data[1], indexed_data[2],indexed_data[5] #getting each Image
    if float(reverse) > 0:
      continue
    image_path.append(os.path.join(datadir, center.strip()))   # appending the centered image
    steering.append(float(indexed_data[3])) # appending the steering
    image_path.append(os.path.join(datadir, right.strip()))
    steering.append(float(indexed_data[3])-0.25)
    image_path.append(os.path.join(datadir, left.strip()))
    steering.append(float(indexed_data[3])+0.25)  # appending the steering.
    # print(image_path,steering)
    # print(float(indexed_data[3]))
    # print(float(indexed_data[3]))
    # indexed_data[3] = float(indexed_data[3][1])-0.25
    # print(float(indexed_data[3]))
    # print(float(indexed_data[3]))
  # Debug: print(image_path)
  # Debug: print(steering)
  image_paths = np.asarray(image_path) # converting it to a np array
  steerings = np.asarray(steering) # converting it to a np array
  return image_paths,steerings

image_paths, steerings = load_img_steering(datadir + '/IMG',data)


# ______________________________ Creating my Training / Validation Variables ______________________________


X_train, X_valid, y_train, y_valid = train_test_split(image_paths,steerings, test_size = 0.2, random_state = 6)# Random State is SEED ( it can be random )
# Selecting my train data and my split data

print("Training Samples: {} \nValid Samples: {}".format(len(X_train),len(X_valid)))

# Debug: print(X_train, " : ", y_train)


# ______________________________ Visualizing if there is a big diffrence between them so that I will have to retrain my model... ______________________________


fig,axes = plt.subplots(1,2,figsize=(12,4))

# Debug: indices = [i for i, x in enumerate(y_train) if x == 1]
# Debug: print(indices)
# print(bins_array)
axes[0].hist(y_train, bins = num_bins,width = 0.05 , color = 'blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins = num_bins,width = 0.05 , color = 'red')
axes[1].set_title('Validation set')
# plt.show()

# New Thing Learnt . The plt.bar Number of Pins has an increase or decreased 1


# ______________________________ Increasing my Data Set using augmentation by panning zooming changing brightness and flipping ______________________________


def img_random_zoom(image):
  zoom = iaa.Affine(scale=(1,1.3))
  image = zoom.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
zoomed_image = img_random_zoom(original_image)
fig, axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[1].imshow(zoomed_image)
axes[1].set_title("Zoomed Image")

# plt.show()


def img_random_pan(image):
  pan = iaa.Affine(translate_percent={"x":(-0.1,0.1) , "y" : (-0.1,0.1)})
  image = pan.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
panned_image = img_random_pan(original_image)
fig, axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[1].imshow(panned_image)
axes[1].set_title("Panned Image")

# plt.show()

def img_random_brightness(image):
  brightness = iaa.Multiply((0.2,1.2))
  image = brightness.augment_image(image)
  return image

image = image_paths[random.randint(0,1000)]
original_image = mpimg.imread(image)
brightness_image = img_random_brightness(original_image)
fig, axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[1].imshow(brightness_image)
axes[1].set_title("brightness Image")

# plt.show()


def img_random_flip(image,steering_angle):
  image = cv2.flip(image,1)
  steering_angle = -steering_angle
  return image,steering_angle

random_index = random.randint(0,1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]
original_image = mpimg.imread(image)
flipped_image , flipped_steering_angle = img_random_flip(original_image,steering_angle)
fig, axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axes[0].imshow(original_image)
axes[0].set_title("Original Image" + str(steering_angle))
axes[1].imshow(flipped_image)
axes[1].set_title("flipped Image" + str(flipped_steering_angle))

# plt.show()
# sys.exit()


def random_augment(image,steering_angle):
  image = mpimg.imread(image)
  if np.random.rand() < main_probability:
    image = img_random_pan(image)

  if np.random.rand() < main_probability:
    image = img_random_zoom(image)

  if np.random.rand() < main_probability:
    image = img_random_brightness(image)

  if np.random.rand() < main_probability:
    image , steering_angle = img_random_flip(image , steering_angle)

  return image,steering_angle


ncol = 2
nrow = 10

fig , axes = plt.subplots(nrow,ncol,figsize = (15,50))
fig.tight_layout()

for i in range(10):
  random_number = random.randint(0,len(image_paths) - 1 )
  random_image = image_paths[random_number]
  random_steering = steerings[random_number]

  original_image = mpimg.imread(random_image)
  augmented_image , augmented_random_steering = random_augment(random_image,random_steering)

  axes[i][0].imshow(original_image)
  axes[i][0].set_title("Original Image, "+ str(random_steering))

  axes[i][1].imshow(augmented_image)
  axes[i][1].set_title("Augmeneted Image , "+ str(augmented_random_steering))

# plt.show()



# ______________________________ Pre Processing Images Function ______________________________




def img_preprocess(img):
    img = img[60:135, :, :]  # trimming
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Because I will use Nvidia Model (The say it is better with yuv)
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Smoothing
    img = cv2.resize(img, (200, 66))  # this size is used by nvidia model arch
    img = img / 255

    return img


# ______________________________ Testing The Diffrence if img preprocessing ______________________________
image = image_paths[100]

original_image = mpimg.imread(image)
preprocessed_image = img_preprocess(original_image)

fix , axes = plt.subplots(1,2,figsize = (15,10))
fig.tight_layout()

print("The Image Size is ",original_image.shape)

axes[0].imshow(original_image)
axes[0].set_title("Original Image")

axes[1].imshow(preprocessed_image)
axes[1].set_title("Preprocessed Image")
# plt.show()


# ______________________________ Creating a Batch For Generating Images While Fitting my model ______________________________


def batch_generator(btimage_paths,btsteering_angle,batch_size,is_training): # in here I am using a subroutines (CHECK IT ) TODO
  while True:
    batch_img = []
    batch_steering = []

    for i in range(batch_size):
      random_index = random.randint(0,len(btimage_paths) - 1)

      if is_training:
        btimg, btsteering = random_augment(btimage_paths[random_index],btsteering_angle[random_index])
      else:
        btimg = mpimg.imread(btimage_paths[random_index])
        btsteering = btsteering_angle[random_index]

      btimg = img_preprocess(btimg)

      batch_img.append(btimg)
      batch_steering.append(btsteering)

    yield (np.asarray(batch_img),np.asarray(batch_steering))


X_train_gen, y_train_gen = next(batch_generator(X_train,y_train,1,1))
X_valid_gen, y_valid_gen = next(batch_generator(X_valid,y_valid,1,0))

fig, axes = plt.subplots(1,2,figsize=(15,10))
fig.tight_layout()

axes[0].imshow(X_train_gen[0])
axes[0].set_title("Training Image " + str(y_train_gen[0]))
axes[1].imshow(X_valid_gen[0])
axes[1].set_title("Validation Image "+ str(y_valid_gen[0]))

# THE MOST IMPORTANT PLT.SHOW
# print(X_train_gen.shape)
# ncol = 2
# nrow = 10
#
# fig , axes = plt.subplots(nrow,ncol,figsize = (15,50))
# fig.tight_layout()
# for i in range(10):
#
#   axes[i][0].imshow(X_train_gen[1])
#   axes[i][0].set_title("Training Image, "+ str(y_train_gen[1]))
#
#   axes[i][1].imshow(X_valid_gen[0])
#   axes[i][1].set_title("Validation Image , "+ str(y_valid_gen[0]))
plt.show()
# ______________________________ PreProcessing my IMG DATA ______________________________
#iterate through all the array and get a new elemnt, so takes array and returns array
# X_train = np.array(list(map(img_preprocess,X_train)))
# X_valid = np.array(list(map(img_preprocess,X_valid)))

# ______________________________ Testing The New preprocessed Data ______________________________
# plt.imshow(X_train[random.randint(0,len(X_train)-1)])
# plt.show()

# print(X_train.shape)# x is 200 heigh is 66  but in array shape it shows [0] [200] , [1] [200]





# ______________________________ Creating My Model ______________________________


# we are using behavioral cloning so, I won't use a typical model. I have to invent my own model !

def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24,(5,5),strides=(2,2), input_shape = (66,200,3),activation = 'elu'))
  model.add(Convolution2D(36,(5,5),strides=(2,2),activation = 'elu'))
  model.add(Convolution2D(48,(5,5),strides=(2,2),activation = 'elu'))
  model.add(Convolution2D(64,(3,3),activation = 'elu'))
  model.add(Convolution2D(64,(3,3),activation = 'elu'))


  model.add(Flatten())

  # model.add(Dropout(0.25))#Testing Purposes


  model.add(Dense(100 , activation = 'elu'))
  model.add(Dropout(0.5))#Testing Purposes

  model.add(Dense(50, activation = 'elu'))
  # model.add(Dropout(0.5))#Testing Purposes

  model.add(Dense(10, activation = 'elu'))
  # model.add(Dropout(0.5))#Testing Purposes

  # model.add(Dense(5, activation = 'elu'))

  model.add(Dense(5, activation = 'elu'))
  model.add(Dense(4, activation = 'elu'))
  model.add(Dense(3, activation = 'elu'))
  model.add(Dense(2, activation = 'elu'))
  model.add(Dense(1))

  # optimizer = Adam(learning_rate = 1e-3) Test
  optimizer = Adam(learning_rate = 1e-4)
  model.compile(loss='mse',optimizer =optimizer, metrics = ['accuracy'])# , metrics = ['accuracy']
  return model

model = nvidia_model()
print(model.summary())
# ______________________________ Fitting The Model ______________________________

# history = model.fit(X_train, y_train,epochs = 30 , validation_data = (X_valid,y_valid), batch_size = 100 , verbose = 1, shuffle = 1)
# history = model.fit(X_train, y_train,epochs = 30 , validation_data = (X_valid,y_valid), batch_size = 50 , verbose = 1, shuffle = 1)
# history = model.fit_generator(batch_generator(X_train,y_train,batch_size=100,is_training = 1),steps_per_epoch= 300, epochs= 5 ,
#                               validation_data=batch_generator(X_valid,y_valid,batch_size = 100,is_training = 0),validation_steps=200,verbose=1,shuffle= 1)
history = model.fit_generator(batch_generator(X_train,y_train,batch_size=main_batches_train,is_training = 1),steps_per_epoch= main_steps_train, epochs= main_number_epochs ,
                              validation_data=batch_generator(X_valid,y_valid,batch_size = main_batches_valid,is_training = 0),validation_steps=main_batches_valid,verbose=1,shuffle= 1)

# ______________________________ Showing the overfit or under fit ______________________________

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epochs')
plt.show()

model.save('newmodel.h5')
# score = model.evaluate(X_valid_gen, y_valid_gen, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

