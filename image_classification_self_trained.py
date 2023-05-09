#%%

%pip install pandas
%pip install matplotlib
%pip install tensorflow

#%%
%pip install scipy

#%%
#%%
# Import all required 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.applications.mobilenet import MobileNet, decode_predictions, preprocess_input
from keras import preprocessing
from tensorflow.keras.preprocessing import image
import keras.backend as K
from keras.layers import Dense,Flatten,GlobalAveragePooling2D,InputLayer,Dropout
import glob
import scipy
#%%

# folder names containing images of the things you want to classify
classes = []
# plug in the path to your data folder
base_path_names = '/home/shinde/Documents/personal_projects/Bene-Tahini/benetech-making-graphs-accessible/train_classes/*'

for i in glob.glob(base_path_names):
    print(i)
    class_name = i.split('/')[-1]
    print(class_name)
    classes.append(class_name)

#%%

base_path = "/home/shinde/Documents/personal_projects/Bene-Tahini/benetech-making-graphs-accessible/train_classes/"
# define the preprocessing function that should be applied to all images
data_gen = preprocessing.image.ImageDataGenerator(   # loads data in batches from disk
    preprocessing_function=preprocess_input,
    # fill_mode='nearest',
    rotation_range=20,                               # rotate image by a random degree between -20 and 20
    # width_shift_range=0.2,                         # shift image horizontally 
    # height_shift_range=0.2,                        # shift image vertically 
    # horizontal_flip=True,                          # randomly flip image horizontally
    zoom_range=0.5,                                  # apply zoom transformation using zoom factor between 0.5 and 1.5
    # shear_range=0.2                                # shear rotates pics, but makes them be in trapezoids (as opposed to squares)
    validation_split=0.2
)


# %%

# a generator that returns batches of X and y arrays
train_data_gen = data_gen.flow_from_directory(      # points to dir where data lives
        directory=base_path,
        class_mode="categorical",
        classes=classes,
        batch_size=20,
        target_size=(224, 224),
    subset='training'
)

#%%

val_data_gen = data_gen.flow_from_directory(
        directory=base_path,
        class_mode="categorical",
        classes=classes,
        batch_size=20,
        target_size=(224, 224),
    subset='validation'
)

#%%

train_data_gen.class_indices

#%%
K.clear_session()
base_model = MobileNet(
    weights='imagenet',
    include_top=False,                          # keep convolutional layers only
    input_shape=(224, 224, 3)
)

#%%
base_model.summary()   
# %%
base_model.trainable = False  # we don't want to train the base model, since this would destroy filters


#%%
len(classes)

#%%

model = keras.Sequential()
model.add(base_model)
model.add(Flatten())  
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(classes), activation='softmax')) # TODO; Final layer with a length of 2, and softmax activation

#%%

model.summary()    

#%%

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss=keras.losses.categorical_crossentropy, #TODO: why not binary x-entropy?
              metrics=[keras.metrics.categorical_accuracy])

# observe the validation loss and stop when it does not improve after 3 iterations
callback = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    #min_delta=0.05,     # the minimum expected change in the metric used in order to be seen as an improvement
    patience=3,         # number of epochs with no improvement needed for the model to stop
    restore_best_weights=True,
    mode='min'
    )

#%%

history = model.fit(train_data_gen,
          verbose=2, 
          callbacks=[callback],
          epochs=20,
          validation_data=val_data_gen
          )

#%%


#%%


plt.plot(history.history['loss'], label = 'Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();


#%%


#%%

#Predict
img = image.load_img('/home/shinde/Documents/personal_projects/Bene-Tahini/benetech-making-graphs-accessible/test/images/000b92c3b098.jpg',target_size=(224,224))
plt.imshow(img);

#%%

img.size
x = np.array(img)
X = np.array([x]) 
X.shape 

#%%

X_preprocess = preprocess_input(X)
pred = model.predict(X_preprocess)
pred.shape

#%%
stats = dict(zip(classes, pred[0]))
max(stats, key=stats.get)


#%%
plt.bar(x = classes, height = pred[0])
plt.xticks(rotation=90, ha='right')
# %%


# preprocess input and predict
preprocesed_image = preprocess_input(X)
plt.imshow(preprocesed_image[0])

#%%

pred_preprocessed = model.predict(preprocesed_image)
pred_preprocessed
# plt.bar(x = classes, height = pred_preprocessed[0])
# plt.xticks(rotation=45, ha='right')
# %%

model.save('models/trained_for_all_objects.h5')
# %%
