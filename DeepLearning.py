import numpy as np 
import pandas as pd
import cv2 
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
import tensorflow as tf
from contextlib import redirect_stdout


# Global variables used on code
f = open("log.txt", "a")
train_df = pd.read_csv('train.csv')
trainImagesPath = ('train_img')
testImagesPath = ('test_img')

# Testing if GPU is connected
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_image(path, pet_id): # Load the first image of a pet
    image = cv2.imread(f'{path}/{pet_id}-1.jpg')
    return image

def load_images(path, pet_id): # Load all images of a pet to a list, return the list and the length of the list
    pictures=glob.glob(f'{path}/{pet_id}-*.jpg')
    return pictures,len(pictures)

def displayImages(imageList, pet_speed): # Gets an ImageList and display all of them maximum 30
    for _,i in enumerate(imageList):
        ax = plt.subplot(6, 2, _ + 1) 
        currentImage=cv2.imread(i)
        plt.imshow(currentImage)
        plt.xlabel(pet_speed)
    plt.show()

# Showing images of pets by ID
petID=train_df['PetID'].iloc[26]
petSpeed=train_df['AdoptionSpeed'].iloc[26]
imageList,imageCount = load_images(trainImagesPath, petID)
# displayImages(imageList, petSpeed)

# Copying dataset to modify
trainCNN_df=train_df.copy()

# Photo amout diff 0
trainCNN_df=trainCNN_df[trainCNN_df.PhotoAmt!=0]

# Creating a new dataset with image and adoption speed
image_df=[]
for index, row in trainCNN_df.iterrows(): 
    imageList,imageCount = load_images(trainImagesPath, row['PetID'])
    if imageCount==0:
        continue
    speed=row['AdoptionSpeed']
    for image in imageList:
        image_df.append([image,speed])
                                                             
f.write(f' we found: {len(image_df)} amount of pictures in the database\n\n\n')
image_df = pd.DataFrame(image_df, columns=['ImageURL','Speed'])

image_df['Speed']=image_df['Speed'].astype(str)

f.write("dataframe:\n{}".format(image_df))

# Appending adoption speed on final dataset to use as target for images classification
# image_df['Speed']=image_df['Speed'].astype(str)
# image_df0=image_df.loc[image_df['Speed']=='0']
# image_df1=image_df.loc[image_df['Speed']=='1']
# image_df2=image_df.loc[image_df['Speed']=='2']
# image_df3=image_df.loc[image_df['Speed']=='3']
# image_df4=image_df.loc[image_df['Speed']=='4']

# image_df_final=image_df0.append([image_df1,image_df2,image_df3,image_df4])
# f.write("Speed value counts:{}\n\n\n".format(image_df_final['Speed'].value_counts()))

# Holdout method
pics=image_df['ImageURL']
label=image_df['Speed']
val_split = 0.20
X_train, X_val, y_train, y_val = train_test_split(pics, label, test_size=val_split,stratify=label)

train_CNN = np.concatenate((X_train, y_train))
val_CNN = np.concatenate((X_val, y_val))

train_CNN=pd.DataFrame(list(zip(X_train, y_train)),
              columns=['ImageURL','Speed'])

val_CNN=pd.DataFrame(list(zip(X_val,y_val)),
              columns=['ImageURL','Speed'])

f.write(f' the length of train set is: {len(train_CNN)},the length of validation set is: {len(val_CNN)}\n\n\n')

datagen=ImageDataGenerator(rescale=1./255,rotation_range=45)

datagen2=ImageDataGenerator(rescale=1./255)

train_generator=datagen.flow_from_dataframe(dataframe=train_CNN, x_col="ImageURL", y_col="Speed", class_mode="categorical", target_size=(112,112), batch_size=32,subset="training",shuffle=True,color_mode="rgb")
valid_generator=datagen2.flow_from_dataframe(dataframe=val_CNN, x_col="ImageURL", y_col="Speed", class_mode="categorical", target_size=(112,112), batch_size=32,subset="validation",shuffle=True,color_mode="rgb")

## Loading VGG16 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(112,112,3))
#base_model.trainable = False ## Not trainable weights

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='sigmoid'),
    layers.Dense(5, activation='softmax')
])


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.05,
        decay_steps=100,
        decay_rate=0.99)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

model.compile(
        loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy']
    )

with open('log.txt', 'a') as file:
    with redirect_stdout(file):
        model.summary()

f.write('\n\n\n')

epochs=100
history=model.fit(train_generator, validation_data=valid_generator, epochs=epochs, batch_size=32)