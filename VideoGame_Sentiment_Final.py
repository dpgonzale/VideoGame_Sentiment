#load dependencies
import pandas as pd
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import TimeseriesGenerator

# Read data and drop unneeded columns
dataset = pd.read_csv('ign.csv').iloc[:, 1:3]  #import relevelant columns from dataset
dataset.fillna(value='', inplace=True)         #replance any blank or nan data with empty string
#print(dataset.columns)

# Check for null or missing data
#dataset.isnull().sum()

# Fill in or create missing data
# dataframe.fillna(value='', inplace=True)

# Extract independent & dependent variables (X & Y)
trainX = dataset.title
trainY = dataset.score_phrase
#trainX
#trainY

# Convert sequence data (strings) into numeric data
tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<UNK>")
tokenizer.fit_on_texts(trainX)
x = tokenizer.texts_to_sequences(trainX)

# Convert data into a matrix array and pad with zeros
totalX = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=15, padding='post', truncating='post')
#totalX

# Convert sequence data (strings) into numeric data
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(trainY)
Y = tokenizer.texts_to_sequences(trainY)
#print(Y.count([11]))

# Convert data into a matrix array
totalY = np.array(Y)
#totalY

# Convert the indices into 11 dimensional vectors
tocatY =  to_categorical(totalY, nb_classes=12)
# Drop first column of zeros
totalY = np.delete(tocatY, 0, 1)
#totalY

# Split data into training, test and validation
trainX, testX, trainY, testY = train_test_split(totalX, totalY, test_size=0.1)

# Build Model
net = tflearn.input_data([None, trainX.shape[1]])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.9)
#net = tflearn.lstm(net, 128, dropout=0.6)
net = tflearn.fully_connected(net, 11, activation='softmax') # relu or softmax
net = tflearn.regression(net, optimizer='adam', learning_rate=.0001, loss='categorical_crossentropy')
#model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path='SavedModels/model.tfl.ckpt')
model = tflearn.DNN(net, tensorboard_verbose=0)

# train the model
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=100, n_epoch=100)
