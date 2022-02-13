import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

targets_numpy = np.array(train.label.values)
features_numpy = np.expand_dims((train.loc[:, train.columns != 'label'].values.astype('float16'))/225, -1)
print(f'inpute shape: {features_numpy.shape}')
print(f'label shape: {targets_numpy.shape}')
encoder = tf.keras.layers.CategoryEncoding(num_tokens = 10, output_mode = 'one_hot')
train_targets_numpy = encoder(targets_numpy[:40000])
print(f'encoder label shape: {targets_numpy.shape}')
print(f'example: {train_targets_numpy[0]}')

features_train = features_numpy[:30000]
targets_train = train_targets_numpy[:30000]

features_valid = features_numpy[30000:40000]
targets_valid = train_targets_numpy[30000:]

features_test = features_numpy[40000:]
targets_test = targets_numpy[40000:]

batch_size = 100
n_iters = 5000
num_epochs = n_iters/(len(features_train)/batch_size)
num_epochs = int(num_epochs)

plt.imshow(np.array(features_numpy[1].reshape(1, 784), dtype = np.float32).reshape(28, 28))
plt.axis('off')
plt.title(str(targets_numpy[1]))
plt.savefig('graph.png')
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = (784, 1)))
model.add(Conv1D(128, 6, strides = 1, activation = 'relu'))
model.add(MaxPool1D(2))
model.add(Conv1D(128, 4, strides = 1, activation = 'relu'))
model.add(MaxPool1D(4))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['acc'])

history = model.fit(features_train, targets_train, epochs = 30, batch_size = 128, use_multiprocessing=True, validation_data = (features_valid, targets_valid))
model.save_weights('models/tensorflow_CNN_model.h5')
def plot(data, label):
    plt.xlabel('epochs')
    plt.ylabel(label[0])
    for l in label:
        plt.title(l + ' vs epochs')
        plt.plot(data[l], label = l)
    plt.legend()
    plt.show()
plot(history.history, label = ['loss', 'val_loss'])
plot(history.history, label = ['acc', 'val_acc'])

model.load_weights('models/tensorflow_CNN_model.h5')
result = pd.DataFrame(columns = ['prediction', 'true'])
predict = model(features_test).numpy()
result['prediction'] = np.argmax(predict, axis = 1)
result['true'] = targets_test
total = len(predict)
correct = (result['prediction'] == result['true']).sum()
print(100 * correct/float(total))

result.to_csv('results/ressults_CNNmodel_tensorflow')
