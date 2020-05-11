"""
FOR LOCAL. 
Read in true positives, negatives, and Majestic Million TN's. 
Break off labels and set hyperparameters, tokenize and sequence query names.
Get dummies for labels, create train/test split. 
Create LSTM, run it, and save out. Save history in end comments. 
"""

import keras as k
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.regularizers import l2
from keras import preprocessing
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import keras_metrics as km
from keras_metrics import binary_precision

#get TP ground truth
tpgt = pd.read_csv("Downloads/gt_full.csv", error_bad_lines=False, lineterminator='\n', header = 0, sep="\t", encoding='utf-8')

#assign bad label
tpgt['label'] = 1

#drop na and dupes
tpgt = tpgt.dropna() #dupe check: tpgt.duplicated(subset='qname', keep='first').sum()
tpgt = tpgt.drop_duplicates(subset = 'qname', keep='first')

#get TN’s + MM
zero_negs = pd.read_csv("Downloads/zero_negs_sample.csv", error_bad_lines=False, lineterminator='\n', header = 0, sep="\t", encoding='utf-8')

#get empty cols for other features
zero_negs['variant'] = 0
zero_negs['label'] = 0

#drop na and dupes
zero_negs = zero_negs.dropna()
zero_negs = zero_negs.drop_duplicates(subset = 'qname', keep='first')


#get equal data samples
zero_negs = zero_negs.sample(500000, replace=False, random_state=100) 
tpgt = tpgt.sample(500000, replace=False, random_state=100) 


#concat dfs, add MM, shuffle and drop na's/dupes again bc OCD
all_data = pd.concat([zero_negs, tpgt])
mm1 = pd.read_csv('Downloads/majestic_million.csv', error_bad_lines=False, header = 0, sep=",", encoding='utf-8', engine = 'python')
mm1['label'] = 0
mm1['variant'] = 0
mm1 = mm1.drop(['GlobalRank', 'TldRank', 'TLD', 'RefSubNets', 'RefIPs', 'IDN_Domain', 'IDN_TLD', 'PrevGlobalRank', 'PrevTldRank', 'PrevRefSubNets', 'PrevRefIPs'], axis=1)
mm1['qname'] = mm1['Domain']
mm1 = mm1.drop(['Domain'], axis=1)
col_order = ['qname', 'variant', 'label']
mm1 = mm1.reindex(columns=col_order)

all_data = pd.concat([all_data, mm1])
all_data = shuffle(all_data)
all_data = all_data.dropna()
all_data = all_data.drop_duplicates(subset = 'qname', keep='first')


#get labels
labels = all_data['label'].copy() #get labels outside data
all_data = all_data.drop(['label'], axis=1) #drop labels from data
all_data = all_data.drop(['variant'], axis=1)


#set some hypers
MAX_NB_WORDS = 500000 #The maximum number of "words" to be used

# Max number of words in seq
MAX_SEQUENCE_LENGTH = 250

# This is fixed value for dim of embed. seems standard ish
EMBEDDING_DIM = 128

#do tokenization of chars in domains
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=True) 

#fit tokenizer
tokenizer.fit_on_texts(all_data['qname'])

char_index = tokenizer.word_index #sanity check
#print('Found %s unique tokens.' % len(word_index))

#actually tokenize queries 
x = tokenizer.texts_to_sequences(all_data['qname'].values)

#pad things to be equal. possibly not necessary re woodbridge et al
x = pad_sequences(x, maxlen=MAX_SEQUENCE_LENGTH)
#print('Shape of data tensor:', x.shape)


#dummy labels
y = pd.get_dummies(labels).values
#print('Shape of label tensor:', y.shape)


#train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)
#print(x_train.shape, y_train.shape)
#print(x_test.shape, y_test.shape)


#lstm
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=x.shape[1]))
model.add(LSTM(44, dropout=0.25, recurrent_dropout=0.25)) 
model.add(Dense(2, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #, km.binary_precision(), km.binary_recall()

epochs = 5
batch_size = 3000 

#history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3)
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=1, validation_data=[x_test, y_test], use_multiprocessing=True)


scores = model.evaluate(x_test, y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model.save("lstm_3.h5")
print("Saved model to disk")
