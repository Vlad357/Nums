import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv', dtype = np.float32)

targets_numpy = train.label.values
features_numpy = train.loc[:, train.columns != 'label'].values/225

features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
                                                                              targets_numpy,
                                                                              test_size = 0.2,
                                                                              random_state = 42)

featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)

featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)

batch_size = 100
n_iters = 5000
num_epochs = n_iters/(len(features_train)/batch_size)
num_epochs = int(num_epochs)

train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

plt.imshow(features_numpy[1].reshape(28, 28))
plt.axis('off')
plt.title(str(targets_numpy[1]))
plt.savefig('graph.png')
plt.show()


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)

        return out


train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

model = CNNModel()

error = nn.CrossEntropyLoss()

learning_rate = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(train)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()

        count += 1

        if count % 50 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                test = Variable(images.view(100, 1, 28, 28))

                outputs = model(test)

                predicted = torch.max(outputs.data, 1)[1]

                total += len(labels)

                correct += (predicted == labels).sum()

            accuracy = 100 * correct / float(total)

            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
    print('epoch: {}  Loss: {}  Accuracy: {} %'.format(epoch + 1, loss.data, int(accuracy)))


def plot(data, labels, colors):
    for i in range(len(data)):
        plt.plot(iteration_list, data[i], color=colors[i])
        plt.xlabel("Number of iteration")
        plt.ylabel(labels[i])
        plt.title(f"ANN: {labels[i]} vs Number of iteration")
        plt.show()


plot(data=(loss_list, accuracy_list),
     labels=('loss', 'accuracy'),
     colors=('red', 'blue'))

torch.save(model, 'models/CNN_torch_model')

model = torch.load('models/CNN_torch_model')

results = pd.DataFrame(columns = ['prediction', 'true'])
total = 0
correct = 0
accuracy = 0
for images, labels in test_loader:
    test = Variable(images.view(100, 1, 28, 28))
    outputs = model(test)
    predicted = torch.max(outputs.data, 1)[1]
    results['prediction'] = predicted
    results['true'] = labels
    total += len(labels)
    correct += (predicted == labels).sum()
    accuracy = 100 * correct / float(total)
print(f'accuracy: {accuracy}')

results.to_csv('results/ressults_CNNmodel_torch')