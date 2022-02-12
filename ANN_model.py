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
n_iters = 10000
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


class ANNModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ANNModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.elu3 = nn.ELU()

        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.tanh2(out)

        out = self.fc3(out)
        out = self.elu3(out)

        out = self.fc4(out)
        return out


input_dim = 28 * 28
hidden_dim = 150
output_dim = 10

model = ANNModel(input_dim, hidden_dim, output_dim)
error = nn.CrossEntropyLoss()

learning_rate = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

count = 0
loss_list = []
iteration_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(-1, 28 * 28))
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
                test = Variable(images.view(-1, 28 * 28))

                outputs = model(test)

                predicted = torch.max(outputs.data, 1)[1]

                total += len(labels)
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / float(total)

            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)
    print('epoch: {}  Loss: {}  Accuracy: {} %'.format(epoch + 1, loss.data, int(accuracy)))
torch.save(model, 'models/ANN_torch_model')


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

model = torch.load('models/ANN_torch_model')

results = pd.DataFrame(columns = ['prediction', 'true'])
total = 0
correct = 0
accuracy = 0
for images, labels in test_loader:
    test = Variable(images.view(-1, 28*28))
    outputs = model(test)
    predicted = torch.max(outputs.data, 1)[1]
    results['prediction'] = predicted
    results['true'] = labels
    total += len(labels)
    correct += (predicted == labels).sum()
    accuracy = 100 * correct / float(total)
print(f'accuracy: {accuracy}')

results.to_csv('results/ressults_ANNmodel_torch')
