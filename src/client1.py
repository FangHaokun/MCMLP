import pickle

import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from phe import paillier
from sklearn.metrics import r2_score

dataframe = pd.read_csv('dataa.csv')
# print(dataframe)

labels = []
features = []

for n in range(0, 437):
    label = dataframe.iloc[n, -1]
    labels.append([label / 4])
    feature = dataframe.iloc[n, 1:-1]
    feature = np.asarray(feature)
    features.append(feature)

features = np.transpose(features)
for i in features:
    maxNum = max(i)
    minNum = min(i)
    for j in range(len(i)):
        i[j] = (i[j] - minNum) / (maxNum - minNum)
features = np.transpose(features)

input = torch.FloatTensor(features[0:300])
label = torch.FloatTensor(labels[0:300])

print(input)
print(label)
# train data
x_data = Variable(input)
y_data = Variable(label)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(17, 17)  # One in and one out
        self.sm = torch.nn.Sigmoid()
        self.linear1 = torch.nn.Linear(17, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = self.sm(y_pred)
        y_pred = self.linear1(y_pred)
        y_pred = self.sm(y_pred)
        return y_pred


# our model
model = Model()

criterion = torch.nn.MSELoss(size_average=False)  # Defined loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Defined optimizer

# Training: forward, loss, backward, step
# Training loop
losses = []
for epoch in range(10000):
    # Forward pass
    y_pred = model(x_data)
    # print(model.linear.weight)
    # Compute loss
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    losses.append(loss.item())
    # Zero gradients
    optimizer.zero_grad()
    # perform backward pass
    loss.backward()
    # update weights
    optimizer.step()

with open('dataa.dat','wb') as fout:
    pickle.dump(losses, fout)


x = []
y = []
# After training
cnt = 0
right = 0
wrong = 0
f = open('a1.dat','w')
for i in range(300, 437):
    test_input = [features[i]]
    test_label = labels[i]
    # print('testinput: ',test_input)
    # print('testlabel: ', test_label)
    hour_var = Variable(torch.Tensor(test_input))
    target = int(test_label[0] * 4)
    pred_res = (model.forward(hour_var).data[0][0].item() * 4)
    print(type(pred_res))
    print("predict (after training)", target, pred_res)
    f.write(f'{target} {pred_res}\n')
    if target == round(pred_res):
        right += 1
    else:
        wrong += 1
    cnt += 1
    # print(test_label)
print('Finally the right/cnt = ', right / cnt)
f.close()