import os
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split

csvdir = r'C:\Users\Zain Ul Ibad\Desktop\aistuff\MovieGenre.csv'
imagedir = r'C:\Users\Zain Ul Ibad\Desktop\aistuff\images'
cleanedcsvpath = 'cleanedcsvset.csv'

###################################3 pure organisation and filtering!!!!!!!!!!!!!! ######################################

df = pd.read_csv(csvdir, encoding = 'ISO-8859-1')  #load original csv file
df['Genres'] = df['Genre'].str.split('|') #split genre into relatively cleaner looking index
allgenres = [g for sublist in df['Genres'].dropna() for g in sublist]  
uniquegenres = sorted(list(set(allgenres)))  #extract unique genres

genredataframe = pd.DataFrame(0, index=df.index, columns= uniquegenres) #convert genre ki list to binary vector and init genres to 0
for idx, genres in enumerate(df['Genres']):   #tis for loop initialises all genres to 1 after iterating thru them
    if isinstance(genres, list):
        for g in genres:
            if g in uniquegenres:
                genredataframe.loc[idx, g] = 1

labelsdataframe = pd.concat([df['imdbId'], genredataframe], axis= 1)  #combine with imdbId
labelsdataframe.to_csv('uncleanedcsvset.csv') #uncleaned csv file, use for debugging

downloadedimages = [f.split('.')[0] for f in os.listdir(imagedir) if f.endswith('.jpg')]  #movie posters downloaded
filteredcsv = labelsdataframe[labelsdataframe['imdbId'].astype(str).isin(downloadedimages)] #remove movies w/o posters
filteredcsv.to_csv(cleanedcsvpath, index=False) #save to new csv file


########################################## preprocessing!!!!! ###############################################

imagesize = (182,268)  #images are 182x268
imagesprocessedsize = 30

def preprocessimage(imagepath):
    image = tf.io.read_file(imagepath)  #read image ki file
    image = tf.image.decode_jpeg(image, channels= 3) #decode image(3 channels means 3 colors(RGB))
    image = tf.image.resize(image, imagesize) #failsafe incase some images arent 182x268
    return image/255.0  #scale pixel values better for stability

def createdataset(dataframe):
    imagepaths = [os.path.join(imagedir, f"{imdbId}.jpg") for imdbId in dataframe['imdbId']] #list of path to poster
    labels = dataframe.iloc[:, 1:].values.astype(np.float32) #skip imdb column and make sure rem columns are 32 bit floats
    return tf.data.Dataset.from_tensor_slices((imagepaths, labels)) #tensorflow dataset of format(imagepath, label)

df = pd.read_csv(cleanedcsvpath) #load cleaned labels
traindataframe, tempdataframe = train_test_split(df, test_size=0.2, random_state= 69420) #0.2(20%) to tempdataframe
validationdataframe, testingdataframe = train_test_split(tempdataframe, test_size=0.5, random_state= 69420) #0.5(50%) to testingdataframe

traindataframe.to_csv('trainlabels.csv', index= False)
validationdataframe.to_csv('validatelabels.csv', index= False)
testingdataframe.to_csv('testinglabels.csv', index= False)

traindataset = createdataset(traindataframe).map(lambda path, label: (preprocessimage(path), label))  #creating tensorflow dataset for training
validationdataset = createdataset(validationdataframe).map(lambda path, label: (preprocessimage(path), label))  #creating tensorflow dataset for validation
testingdataset = createdataset(testingdataframe).map(lambda path, label: (preprocessimage(path), label))  #creating tensorflow dataset for testing