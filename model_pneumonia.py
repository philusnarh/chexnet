# Import packages 
import os, glob 
import sys
import datetime
import random
import math
import numpy as np
import csv
import matplotlib.pyplot as plt
from skimage import  measure 
from skimage.transform import resize
import tensorflow as tf
from tensorflow import keras 
import pydicom
from imgaug import augmenters as iaa
import pandas as pd 
from keras.layers import Dropout

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from tensorflow.python.client import device_lib
# To check gpu
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
# Check gpu
get_available_gpus()

#chmod -R 0777 /home/ubuntu/Kaggle_Module_2_Data/stage_1_detailed_class_info.csv

#### function to create a subsample from main dataset #######
def dat_subsample(csvpath,sample_size=2000):
    ### return list of images that were selected, list of images not selected 
    ### the dataframe of image selected and the dataframe of images not selected
    pneumonia = 8964
    notpeunomia = 11500+8525
    total_im = pneumonia+notpeunomia
    #putting a seed number so that everyone has the same list of random numbers 
    random.seed(1023)
    info_file = pd.read_csv(csvpath+'stage_1_detailed_class_info.csv')
    #selecting the 3 classes and saving them in different dataframes
    pneu_df = info_file[(info_file['class']=='Lung Opacity')]
    notnorm_df = info_file[(info_file['class']=='No Lung Opacity / Not Normal')]
    norm_df = info_file[(info_file['class']=='Normal')]
    # creating 3 random number list
    #pneu_ranlst = nprand.randint(0,pneu_df.shape[0],sample_size//2)
    #notnorm_ranlst = nprand.randint(0,notnorm_df.shape[0],sample_size//4)
    #norm_ranlst = nprand.randint(0,norm_df.shape[0],sample_size//4)
    pneu_ranlst = random.sample(range(pneu_df.shape[0]),sample_size//2)
    notnorm_ranlst = random.sample(range(notnorm_df.shape[0]),sample_size//4)
    norm_ranlst = random.sample(range(norm_df.shape[0]),sample_size//4)
    #randomly selected image dataframe 
    sel_pneu_lst = pneu_df.iloc[pneu_ranlst].index.values.tolist()
    sel_notnorm_lst = notnorm_df.iloc[notnorm_ranlst].index.values.tolist()
    sel_norm_lst = norm_df.iloc[norm_ranlst].index.values.tolist()
    #joining all the list to get a complete list of images
    main_sel_lst = []
    main_sel_lst.extend(sel_pneu_lst)
    main_sel_lst.extend(sel_notnorm_lst)
    main_sel_lst.extend(sel_norm_lst)

    #getting the list of selected images and getting the corresponding dataframe
    selected_image_lst = info_file.iloc[main_sel_lst]['patientId'].values.tolist()
    selected_image_df = info_file.iloc[main_sel_lst]

    #getting the not selected list and dataframe - could be use for cross validation
    notselected_image_df = info_file.loc[info_file.index.difference(selected_image_df.index), ]
    notselected_image_lst = notselected_image_df['patientId'].values.tolist()
    return selected_image_lst,notselected_image_lst,selected_image_df,notselected_image_df
###########################################################################################

# Create a subsample from main dataset
selected_image_lst,notselected_image_lst,selected_image_df,notselected_image_df=dat_subsample('/home/ubuntu/Kaggle_Module_2_Data/',sample_size=2000)

a=list(selected_image_df['patientId'].values)
random.shuffle(a)

image_list = []
for char in a: 
    char = char+'.dcm' 
    image_list.append(char)
    
    
    

# Loading the pneumonia locations
pneumonia_locations = {}
# load train labels
with open(os.path.join('/home/ubuntu/Kaggle_Module_2_Data/stage_1_train_labels.csv'), mode = 'r') as infile:
  # open reader
    reader = csv.reader(infile)
    # skip header
    next(reader, None)
    # loop through rows
    for rows in reader:
        # retrieve information
        filename = rows[0]
        location = rows[1:5]
        pneumonia = rows[5]
        # if row contains pneumonia add label to dictionary
        # which contains a list of pneumonia locations per filename
        if pneumonia == '1':
            # convert string to float to int
            location = [int(float(i)) for i in location]
            # save pneumonia location in dictionary
            if filename in pneumonia_locations:
                pneumonia_locations[filename].append(location)
            else:
                pneumonia_locations[filename] = [location]
  
  

#chmod -R 0777 /home/ubuntu/Kaggle_Module_2_Data/stage_1_train_images
# Load filenames and partition into train and validation set
#folder = '/home/ubuntu/Kaggle_Module_2_Data/stage_1_train_images'
#filenames = os.listdir(folder)
# random.shuffle(image_list)
# split into train and validation filenames
n_valid_samples = 1500
valid_filenames = image_list[n_valid_samples:]
train_filenames = image_list[:n_valid_samples]
print('n train samples', len(train_filenames))
print('n valid samples', len(valid_filenames))
n_train_samples = len(image_list) - n_valid_samples

class generator(keras.utils.Sequence):
    
    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32, image_size=256, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()
        
    def __load__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # create empty mask
        msk = np.zeros(img.shape)
        # get filename without extension
        filename = filename.split('.')[0]
        # if image contains pneumonia
        if filename in pneumonia_locations:
            # loop through pneumonia
            for location in pneumonia_locations[filename]:
                # add 1's at the location of the pneumonia
                x, y, w, h = location
                msk[y:y+h, x:x+w] = 1
        # if augment then horizontal flip half the time
        if self.augment and random.random() > 0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)
        # resize both image and mask
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk
    
    def __loadpredict__(self, filename):
        # load dicom file as numpy array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        # resize image
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        # add trailing channel dimension
        img = np.expand_dims(img, -1)
        return img
        
    def __getitem__(self, index):
        # select batch
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]
        # predict mode: return images and filenames
        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create numpy batch
            imgs = np.array(imgs)
            return imgs, filenames
        # train mode: return images and masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip images and masks
            imgs, msks = zip(*items)
            # create numpy batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks
        
    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)
        
    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames) / self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames) / self.batch_size)
        
BATCH_SIZE = 35
IMAGE_SIZE = 256
drop_prob_1 = 0.00
drop_prob_2 = 0.50


def create_downsample(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 1, padding='same', use_bias=False)(x)
    x = keras.layers.MaxPool2D(2)(x)
#    x = keras.layers.Dropout(drop_prob_1)(x)
    return x

def create_resblock(channels, inputs):
    x = keras.layers.BatchNormalization(momentum=0.9)(inputs)
    x = keras.layers.LeakyReLU(0)(x)
 #   x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
 #   x = keras.layers.Dropout(drop_prob_2)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(x)
    return keras.layers.add([x, inputs])

def create_network(input_size, channels, n_blocks=2, depth=4):
    # input
    inputs = keras.Input(shape=(input_size, input_size, 1))
    x = keras.layers.Conv2D(channels, 3, padding='same', use_bias=False)(inputs)
    # residual blocks
    for d in range(depth):
        channels = channels * 2
        x = create_downsample(channels, x)
        for b in range(n_blocks):
            x = create_resblock(channels, x)
    # output
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(256, 1, activation=None)(x)
    x = keras.layers.Dropout(drop_prob_2)(x)
    x = keras.layers.Conv2D(256, 1, activation=None)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2DTranspose(128, (8,8), (4,4), padding="same", activation=None)(x)
    x = keras.layers.BatchNormalization(momentum=0.9)(x)
    x = keras.layers.LeakyReLU(0)(x)
    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)
    outputs = keras.layers.UpSampling2D(2**(depth-2))(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# define iou or jaccard loss function
def iou_loss(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)
    return 1 - score

# combine bce loss and iou loss
def iou_bce_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) + 0.5 * iou_loss(y_true, y_pred)

# mean iou as a metric
def mean_iou(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersect = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    smooth = tf.ones(tf.shape(intersect))
    return tf.reduce_mean((intersect + smooth) / (union - intersect + smooth))

# create network and compiler
model = create_network(input_size=IMAGE_SIZE, channels=32, n_blocks=2, depth=4)
model.compile(optimizer='adam',
              loss=iou_bce_loss,
              metrics=['accuracy', mean_iou])

# cosine learning rate annealing
def cosine_annealing(x):
    lr = 0.001
    epochs = 20
    return lr*(np.cos(np.pi*x/epochs)+1.)/2
learning_rate = tf.keras.callbacks.LearningRateScheduler(cosine_annealing)

# create train and validation generators
folder = '/home/ubuntu/Kaggle_Module_2_Data/stage_1_train_images'
train_gen = generator(folder, train_filenames, pneumonia_locations, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=True, augment=True, predict=False)
valid_gen = generator(folder, valid_filenames, pneumonia_locations, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, shuffle=False, predict=False)

print(model.summary())

def model_fit():
    start_time = datetime.datetime.now()
    hist = model.fit_generator(train_gen, validation_data=valid_gen, callbacks=[learning_rate], epochs=25, shuffle=True)
    end_time = datetime.datetime.now()
    delta = (end_time - start_time).seconds
    print('Training Done..., Time Cost: %d' % ((end_time - start_time).seconds))
    return hist, delta
history, delta = model_fit()
#model.save('model_pneumonia.h5') 

 
plt.figure(figsize=(12,4))
plt.subplot(131)
plt.plot(history.epoch, history.history["loss"],'bo', label="Train loss")
plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
plt.legend()
plt.subplot(132)
plt.plot(history.epoch, history.history["acc"],'bo', label="Train accuracy")
plt.plot(history.epoch, history.history["val_acc"], label="Valid accuracy")
plt.legend()
plt.subplot(133)
plt.plot(history.epoch, history.history["mean_iou"],'bo', label="Train iou")
plt.plot(history.epoch, history.history["val_mean_iou"], label="Valid iou")
plt.legend()
# plt.show()
now = datetime.datetime.now()

plt.savefig(str(history.history["val_mean_iou"][-1])+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.png')
#!chmod -R 0777 /home/ubuntu/Kaggle_Module_2_Data/stage_1_test_images

# load and shuffle filenames
folder = '/home/ubuntu/Kaggle_Module_2_Data/stage_1_test_images'
test_filenames = os.listdir(folder)
print('n test samples:', len(test_filenames))

# create test generator with predict flag set to True
test_gen = generator(folder, test_filenames, None, batch_size=16, image_size=IMAGE_SIZE, shuffle=False, predict=True)

# create submission dictionary
submission_dict = {}
# loop through testset
for imgs, filenames in test_gen:
    # predict batch of images
    preds = model.predict(imgs)
    # loop through batch
    for pred, filename in zip(preds, filenames):
        # resize predicted mask
        pred = resize(pred, (1024, 1024), mode='reflect')
        # threshold predicted mask
        comp = pred[:, :, 0] > 0.5
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            # proxy for confidence score
            conf = np.mean(pred[y:y+height, x:x+width])
            # add to predictionString
            predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' '
        # add filename and predictionString to dictionary
        filename = filename.split('.')[0]
        submission_dict[filename] = predictionString
    # stop if we've got them all
    if len(submission_dict) >= len(test_filenames):
        break
        
print("Done predicting...")
        
# save dictionary as csv file
sub = pd.DataFrame.from_dict(submission_dict,orient='index')
sub.index.names = ['patientId']
sub.columns = ['PredictionString']
#sub.to_csv('submission.csv')

def create_submission():
    now = datetime.datetime.now()
    submission_csv = 'submission_'+str(history.history["val_mean_iou"][-1])+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print ('Creating submission: ', submission_csv)
    sub.to_csv(submission_csv)
    print('Submission: '+submission_csv+' created.')
create_submission()
