import numpy as np
import os
import utils

from CNNModel import CNN
from torch import optim
import torch.nn as nn
import torch

# There are three versions of MNIST dataset
dataTypes = ["digits-normal.mat", "digits-scaled.mat", "digits-jitter.mat"]

# Accuracy placeholder
accuracy = np.zeros(len(dataTypes))
trainSet = 1
testSet = 2

for i in range(len(dataTypes)):
    dataType = dataTypes[i]

    # Load data
    path = os.path.join("..", "data", dataType)
    data = utils.loadmat(path)
    print("+++ Loading dataset: {} ({} images)".format(dataType, data["x"].shape[2]))

    # Organize into numImages x numChannels x width x height
    x = data["x"].transpose([2, 0, 1])
    x = np.reshape(x, [x.shape[0], 1, x.shape[1], x.shape[2]])
    y = data["y"]
    # Convert data into torch tensors
    x = torch.tensor(x).float()
    y = torch.tensor(y).long()  # Labels are categorical

    # Define the model (implement this)
    model = CNN()
    model.train()
    # print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training
    xTrain = x[data["set"] == trainSet, :, :, :]
    yTrain = y[data["set"] == trainSet]

    # Loop over training data in some batches
    numTrain = xTrain.shape[0]
    numIters = 1000  # Do not adjust this
    batchSize = 32  # Do not adjust this

    for iter in range(numIters):
        mini_batch = torch.randint(0, numTrain, (batchSize,))
        this_x = xTrain[mini_batch]
        this_y = yTrain[mini_batch]

        optimizer.zero_grad()

        output = model(this_x)
        loss = loss_function(output, this_y)

        loss.backward()
        optimizer.step()

        if (iter + 1) % 100 == 0:
            print(f"Iteration [{iter + 1}/{numIters}], Loss: {loss.item():.4f}")

    # Test model
    xTest = x[data["set"] == testSet, :, :, :]
    yTest = y[data["set"] == testSet]
    yPred = np.zeros(yTest.shape[0])

    model.eval()

    with torch.no_grad():
        for j in range(yTest.shape[0]):
            sample = xTest[j].unsqueeze(0)

            output = model(sample)
            prediction = torch.argmax(output, dim=1).item()

            yPred[j] = prediction

    # Convert back to numpy arrays
    yTest = yTest.numpy()

    # pdb.set_trace()
    (acc, conf) = utils.evaluateLabels(yTest, yPred, False)
    print("Accuracy [testSet={}] {:.2f} %\n".format(testSet, acc * 100))
    accuracy[i] = acc

# Print the results in a table
print("+++ Accuracy Table [trainSet={}, testSet={}]".format(trainSet, testSet))
print("--------------------------------------------------")
print("dataset\t\t\t", end="")
print("{}\t".format("cnn"), end="")
print()
print("--------------------------------------------------")
for i in range(len(dataTypes)):
    print("{}\t".format(dataTypes[i]), end="")
    print("{:.2f}\t".format(accuracy[i] * 100))

# Once you have optimized the hyperparameters, you can report test accuracy
# by setting testSet=3. Do not optimize your hyperparameters on the
# test set. That would be cheating.
