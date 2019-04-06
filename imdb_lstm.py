'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
# Notes
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.utils import plot_model
from adam import Adam
from aadam import AAdam
from sgd import SGD
from asgd import ASGD
from adagrad import Adagrad
from aadagrad import AAdagrad
import numpy as np
import pandas as pd

max_features = 5000
# cut texts after this number of words (among top max_features most common words)
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

plot_model(model, to_file='model_imdb.png',show_shapes=True)

results_acc = []
result_acc= []
results_loss = []
result_loss = []
test_acc_results = []
test_loss_results = []
l= [Adam(lr=0.001,amsgrad = True), AAdam(lr=0.001,amsgrad = True),Adam(lr=0.001,amsgrad = False), AAdam(lr=0.001,amsgrad = False),Adagrad(),AAdagrad(),SGD(),ASGD() ] #, Adam(lr=0.001, amsgrad = True), AAdam(lr=0.001, amsgrad = True)]

for opt in l:

    model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    #model.save_weights('initial_weights_imdb.h5')
    model.load_weights('initial_weights_imdb.h5')
    initial_weights = model.get_weights()
    result_acc = []
    result_loss = []
    test_loss = []
    test_acc = []
    for i in range (2):
        model.set_weights(initial_weights)
        result_acc_e = []
        result_loss_e = []
        test_acc_e = []
        test_loss_e = []
        for j in range (10):
            history = model.fit(x_train, y_train,batch_size=batch_size,epochs=1,verbose=0)
            '''if j % 2 == 0 :
                test_loss_j, test_acc_j = model.evaluate(x_test, y_test)
                test_acc_e.append(test_acc_j)
                test_loss_e.append(test_loss_j)'''
            result_acc_e.append(history.history['acc'][0])
            result_loss_e.append(history.history['loss'][0])
        test_loss.append(test_loss_e)
        test_acc.append(test_acc_e)
        result_acc.append(result_acc_e)
        result_loss.append(result_loss_e)
    print("##### NEW OPTIMIZER #####")
    print(opt)
    print(np.mean(result_acc,axis=0))
    print(np.mean(result_loss,axis=0))
    print(np.mean(test_acc,axis=0))
    print(np.mean(test_loss,axis=0))
    results_acc.append(np.mean(result_acc,axis=0))
    results_loss.append(np.mean(result_loss,axis=0))
    test_acc_results.append(np.mean(test_acc,axis=0))
    test_loss_results.append(np.mean(test_loss,axis=0))


df = pd.DataFrame(results_acc)
df.to_csv("results/imdb_acc_train_lstm.csv")
df = pd.DataFrame(results_loss)
df.to_csv("results/imdb_loss_train_lstm.csv")
df = pd.DataFrame(test_acc_results)
df.to_csv("results/imdb_acc_test_lstm.csv")
df = pd.DataFrame(test_loss_results)
df.to_csv("results/imdb_loss_test_lstm.csv")
