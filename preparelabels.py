import os
import pandas as pd
import numpy as np
import tensorflow as tf
# import keras
# from keras import models, layers

imagedir = r'C:\Users\Zain Ul Ibad\Desktop\aistuff\images'

########################### splitting and verification!!!!!!!!!!!!! ##################################################

traininglabels = pd.read_csv('trainlabels.csv')
validationlabels = pd.read_csv('validatelabels.csv') 
testinglabels = pd.read_csv('testinglabels.csv')

assert not traininglabels.empty, "training labels are empty" #throws error msg if empty training labels
assert not validationlabels.empty, "validation labels are empty" #same but for validation labels
assert not testinglabels.empty, "testing labels are empty" #same but for testing labels

print("label shapes: \n")
print(f"Train: {traininglabels.shape}")
print(f"Validation: {validationlabels.shape}")
print(f"Testing: {testinglabels.shape}")

############################ extraction!!!!!!!!!!!!!!!!!!!!! #######################################################

y_train = traininglabels.iloc[:, 1:].values.astype('float32')
y_validation = validationlabels.iloc[:, 1:].values.astype('float32')
y_testing = testinglabels.iloc[:, 1:].values.astype('float32')

print("sample movie label: \n")
print(y_train[0])

############################# tensorflow dataset creation!!!!!!!!!!!!!!!! #############################################

def createverifieddataset(dataframe):
    imagepaths = [os.path.join(imagedir, f"{imdbId}.jpg") for imdbId in dataframe['imdbId']]  #get image path
    labels = dataframe.iloc[:, 1:].values.astype('float32') #get labels
    if len(imagepaths) != len(labels):  #check if image path and image labels are same
        raise ValueError(f"{len(imagepaths)} and {len(labels)} are not same in length")
    
    return tf.data.Dataset.from_tensor_slices((imagepaths, labels))

############################# preparation!!!!!!!!!!!!!!!!!!!! ########################################

noofgenres = y_train.shape[1] #[1] means going into second dimension
cnnmodel = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(182,268,3)),   ##define input shape
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), #image extraction: no of filters(32), kernel size(3x3)
    tf.keras.layers.MaxPooling2D((2,2)), #2,2 halves spatial dimensions
    tf.keras.layers.Flatten(),  #2D to 1D!!!!!
    tf.keras.layers.Dense(128, activation='relu'), #learning, 128 = no of neurons, activation='relu' is nonlinear transfromation
    tf.keras.layers.Dense(noofgenres, activation='sigmoid') #multiple label output, sigmoid gives independent probabilities
])

cnnmodel.compile(optimizer= 'adam',  #adaptive learning algorithm 
                 loss= 'binary_crossentropy',  #loss across all outputs
                 metrics= ['accuracy', tf.keras.metrics.Precision(name= 'Precision')] #measures accuracy and precision
)

print("model summary: \n")
cnnmodel.summary()

################################ verification!!!!!!!!!!!!!!!!! ########################################################

sampledataset = createverifieddataset(traininglabels)
# def load_and_preprocess(image_path, label):
#     image = tf.io.read_file(image_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     image = tf.image.resize(image, [182, 268])
#     return image, label

# sampledataset = sampledataset.map(load_and_preprocess)

sampleimage, samplelabel = next(iter(sampledataset))
# print("\n=== Debug Info ===")
# print("Image shape:", sampleimage.shape)  # Should be (182, 268, 3)
# print("Label shape:", samplelabel.shape)
# print("Label values:", samplelabel.numpy())
# print("Image min/max:", tf.reduce_min(sampleimage).numpy(), 
#                        tf.reduce_max(sampleimage).numpy())
print(f"sample image shape: {sampleimage.shape} \n")
print(f"sample label: {samplelabel.shape} \n")