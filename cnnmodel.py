import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
# import keras
# from keras import layers, models, regularizers
import numpy as np
import pandas as pd

imagedir = r'C:\Users\Zain Ul Ibad\Desktop\aistuff\images'
################################ architecting!!!!!!!!!!!! ###########################################################

def createcnn(inputshape, noofgenres):
    cnnmodel = models.Sequential([
        layers.Input(shape= inputshape, name='input_layer'),  #input layer(raw movie poster images)

        layers.Conv2D(32, (3,3), activation='relu', name='conv1'), #convolution block 1
        layers.MaxPooling2D((2,2), name='pool1'),

        layers.Conv2D(64, (3,3), activation='relu', name='conv2'), #conv block 2
        layers.MaxPooling2D((2,2), name='pool2'),

        layers.Conv2D(128, (3,3), activation='relu', name='conv3'), #conv block 3
        layers.MaxPooling2D((2,2), name='pool3'),

        layers.Flatten(name='flatten'), #flatten layer
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='dense1'), #dense layer
        layers.Dropout(0.5, name='dropout1'), #random deactivation for regularisation
        layers.Dense(noofgenres, activation='sigmoid', name='output') #output layer
    ])
    return cnnmodel

cnnmodel = createcnn(inputshape=(182,168,3), noofgenres=20) #182,168 as image dimesnions are 182x168

############################### compiling!!!!!!!!!!!! ##############################################################

cnnmodel.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss= 'binary_crossentropy', #measures error bw predicted probabilites and true labels
    metrics= [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

############################# pipeline preparing!!!!!!! #######################################################

def preprocessimage(imagepath, label):
    image = tf.io.read_file(imagepath)  #load image
    image = tf.image.decode_jpeg(image, channels=3) #decode image
    image = tf.image.resize(image, [182,168])  #resize if needed taake saare images in same dimensions
    image = tf.cast(image, tf.float32)/255.0  
    return image, label

def createdataset(dataframe):
    imagepaths = [os.path.join(imagedir, f"{id}.jpg") for id in dataframe['imdbId']]
    labels = dataframe.iloc[:, 1:].values.astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((imagepaths, labels)) \
        .map(preprocessimage, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(32) \
        .prefetch(tf.data.AUTOTUNE)

trainingdataframe = pd.read_csv('cleanedtrainlabels.csv')
validationdataframe = pd.read_csv('cleanedvalidatelabels.csv')
testingdataframe = pd.read_csv('cleanedtestinglabels.csv')

#creating datasets
traindataset = createdataset(trainingdataframe)
validatedataset = createdataset(validationdataframe)
testingdataset = createdataset(testingdataframe)

########################## training!!!!!!!!!!!! #####################################################################

earlystopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  #tracks if validation loss is occuring
    patience=5,
    restore_best_weights=True
)

modeltraining = cnnmodel.fit(
    traindataset,  #freshly made trained data
    validation_data=validatedataset,    #validation data
    epochs=50,   #max training epochs
    callbacks=[earlystopping]
)

######################## regularising!!!!!!!!! #####################################################################

dataaugmentationlayers = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),  #randomly flip layers horizontally
    layers.RandomRotation(0.1),   #rotate by 10%(0.1)
    layers.RandomZoom(0.1)  #zoom in/out by 10%
])

def augmentedpreprocessing(imagepath, label):
    image = tf.io.read_file(imagepath)
    image = dataaugmentationlayers(image)  #augmentations applied
    image = tf.image.decode_jpeg(image, channels=3)   #these steps load image without resizing
    image = tf.image.resize(image, [182,268])  #image resized to 182x268(if they havent already)
    image = image / 255.0
    return image, label

augmentationtrainingdataset = createdataset(trainingdataframe).map(   #creates augmented dataset
    lambda x,y: (dataaugmentationlayers(x), y),
    num_parallel_calls=tf.data.AUTOTUNE   #autotune prevents bottlenecking
)

############################## evaluating!!!!!!!!!! ########################################################

testresults = cnnmodel.evaluate(testingdataset)
print(f"\nTest Metrics:")
print(f"Loss: {testresults[0]:.3f}")  #up to 3dp
print(f"Accuracy: {testresults[1]:.3f}")
print(f"Precision: {testresults[2]:.3f}")
print(f"Recall: {testresults[3]:.3f}")

y_true = testingdataframe.iloc[:, 1:].values   #true genre labels(except for imdbId)
y_predictions = cnnmodel.predict(testingdataset) > 0.5  #threshold predictions

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y_true, y_predictions, target_names=testingdataframe.columns[1:]))

############################ deploying!!!!!!!!!!!!!! ###########################################################

cnnmodel.save("movegenreclassifier.h5")

def predictgenres(imagepath, threshold=0.5):
    image = tf.io.read_file(imagepath)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [182, 268])
    image = tf.expand_dims(image/255.0, axis=0) 
    predictions = cnnmodel.predict(image)  #returns probabilities
    return (predictions > threshold).astype(int)[0]  #converts probs to binary predictions

def predictanddisplay(image_path):
    preds = predictgenres(image_path)
    genres = testingdataframe.columns[1:].tolist()
    predicted = [genres[i] for i, val in enumerate(preds) if val]
    print(f"\nPredicted Genres for {image_path}:")
    print(", ".join(predicted) if predicted else "No genres predicted")

predictanddisplay(r"C:\Users\Zain Ul Ibad\Desktop\bruno.jpeg") 