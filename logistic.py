from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import layers
from adam import Adam
from aadam import AAdam
from adamW import AdamW
from sgd_cust import SGDCust
import numpy as np 
import pandas as pd

adamw = AdamW(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., weight_decay=0.025, batch_size=1, samples_per_epoch=1, epochs=1)

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
network = models.Sequential()
network.add(layers.Dense(10, activation='softmax', input_shape=(28 * 28,)))

results_acc = []
result_acc= []
results_loss = []
result_loss = []
test_results = []
test_result = []
l= [Adam(amsgrad = False), Adam(amsgrad = True), AAdam(amsgrad = False), AAdam(amsgrad = True)]

for opt in l:

    network.compile(optimizer= opt,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    #network.save_weights('initial_weights_log.h5')
    network.load_weights('initial_weights_log.h5')
    initial_weights = network.get_weights()
    result_acc = []
    result_loss = []
    test_result = []
    for i in range (20):
        network.set_weights(initial_weights)
        history = network.fit(train_images, train_labels, epochs=20, batch_size=128,verbose=0)
        test_loss, test_acc = network.evaluate(test_images, test_labels)
        test_result.append([test_loss, test_acc])
        result_acc.append(history.history['acc'])
        result_loss.append(history.history['loss'])
    print(np.mean(result_acc,axis=0))
    print(np.mean(result_loss,axis=0))
    print(np.mean(test_result,axis=0))
    results_acc.append(np.mean(result_acc,axis=1))
    results_loss.append(np.mean(result_loss,axis=1))
    test_results.append(np.mean(test_result,axis=1))

df = pd.DataFrame(results_acc)
df.to_csv("acc_train_log.csv")
df = pd.DataFrame(results_loss)
df.to_csv("loss_train_log.csv")
df = pd.DataFrame(test_results)
df.to_csv("results_test_log.csv") 



