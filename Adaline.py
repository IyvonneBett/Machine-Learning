from time import sleep
import pandas as pd

# initialize
learning_rate = 0.2
weights = [0.2] * 2
bias = 0.2


def dot_product(values, weights):
    return sum((value * weight) + bias for value, weight in zip(values, weights))


# Handle data
df = pd.read_csv(r"/Users/mac/PycharmProjects/kmeans/data/Sonar.csv")
dataset = df.astype(float).values.tolist()
print(len(dataset[0]))
training_set = [row[0:-1] for row in dataset]
targets = [row[-1] for row in dataset]
print(len(training_set))
print(len(targets))
print('-' * 60)
sleep(2)

lastEpoch = None
currentEpoch = None
hasConverged = False

while lastEpoch is None or lastEpoch != currentEpoch:
    lastEpoch = currentEpoch
    currentEpoch = []
    error_count = 0
    for inputVector, target in zip(training_set, targets):
        output = dot_product(inputVector, weights)
        error = (target - output)
        newWeights = [oldWeight + learning_rate * error * x for (oldWeight, x) in zip(weights, inputVector)]
        bias = bias + (learning_rate * error)
        print(error)
        print('-' * 60)
        # if error == 0 :
        #     hasConverged = True
        #     break
        # else:
        #     print(weights,newWeights, sep="\n", end="\n\n")
        #     weights = newWeights.copy()
        currentEpoch.append(newWeights)
#print(lastEpoch)
print('-' * 60)
print(currentEpoch)
print('-' * 60)

print("Yay")
