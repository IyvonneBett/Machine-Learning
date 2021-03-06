import os
import sys
import numpy as np
sys.path.append(os.getcwd())

import HeartData

os.system('clear')

def distanceBetween(point1, point2):
    # ignore label
    distance = sum(map(lambda x, y: (x - y) ** 2, point1[:-1], point2[:-1]))
    return distance


def averageOfPoints(cluster):
    return tuple(sum(point) / len(point) for point in zip(*cluster))


def newCentroidsFrom(clusters):
    return [averageOfPoints(cluster) for cluster in clusters]


def kMeansCluster(data, k):
    count = len(data)
    previous = None
    currentClusters = None

    # take first k values as centroids
    centroids = data[:k]

    loops = 0

    # repeat until stable
    while (previous == None or previous != currentClusters):
        previous = currentClusters
        distances = [[] for y in centroids]
        for datum in data:
            index = 0
            for centroid in centroids:
                distances[index].append(distanceBetween(centroid, datum))
                index += 1

        currentClusters = [[] for _ in range(k)]
        transposedDistances = np.transpose(distances)

        for index in range(count):
            minimum = np.argmin(transposedDistances[index])
            currentClusters[minimum].append(data[index])

        centroids = newCentroidsFrom(currentClusters)
        loops = loops + 1
    print(loops)
    return currentClusters


data = HeartData.loadData()

k = 2

clusters = kMeansCluster(data=data,
                         k=k)

print('The cluster found are ')
for cluster in clusters:
    # print(cluster)
    print(len(cluster))
