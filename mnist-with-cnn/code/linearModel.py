import numpy as np


default_params = {'lambda': 0.01, 'maxiter': 200, 'eta': 0.001, 'addBias': True}


def train(x, y, param=default_params):

    return multiclassLRTrain(x, y, param)


def predict(model, x):
    if model["addBias"]:
        x = np.append(x, np.ones((1, x.shape[1])), axis=0)
    prob = softmax(np.matmul(model["w"], x))
    yid = np.argmax(prob, axis=0)
    ypred = model["classLabels"][yid]
    return ypred


def multiclassLRTrain(x, y, param):
    if param["addBias"]:
        x = np.append(x, np.ones((1, x.shape[1])), axis=0)

    classLabels = np.unique(y)
    numClass = classLabels.shape[0]
    numFeats = x.shape[0]
    numData = x.shape[1]

    verboseOutput = False
    if verboseOutput:
        print(
            "Multiclass LR Train: {} class, {} features, {} data points".format(
                numClass, numFeats, numData
            )
        )

    # Generate one-hot labels
    trueProb = np.zeros((numClass, numData))
    for c in range(numClass):
        isClass = y == classLabels[c]
        trueProb[c, isClass] = 1

    # Initialize weights randomly (Implement gradient descent)
    model = {}
    model["w"] = np.random.randn(numClass, numFeats) * 0.01
    model["classLabels"] = classLabels
    model["addBias"] = param["addBias"]

    for iter in range(param["maxiter"]):
        prob = softmax(np.matmul(model["w"], x))
        if verboseOutput:
            objective = np.sum(np.log(np.sum(trueProb * prob, axis=0))) - param[
                "lambda"
            ] * np.sum(model["w"] ** 2)
            print("Iter: {} objective {:.4f}".format(iter, objective))
        delta = trueProb - prob
        gradientLoss = np.matmul(delta, np.transpose(x))
        model["w"] = (1 - param["eta"] * param["lambda"]) * model["w"] + param[
            "eta"
        ] * gradientLoss

    return model


def softmax(z):
    sz = np.exp(z)
    colsum = np.sum(sz, axis=0)
    sz = np.matmul(sz, np.diag(1 / colsum))
    return sz
