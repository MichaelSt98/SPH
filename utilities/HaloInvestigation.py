import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[0]-p2[0])**2 + (p1[0]-p2[0])**2)**(1/2.)

if __name__ == '__main__':

    path = "../output/"
    fileToPlot = "{}particle_00010.txt".format(path)
    numProcesses = 2

    df = pd.read_csv(fileToPlot, delimiter=";", skiprows=numProcesses+1)
    connect = False

    ####################################################################
    print(df.head())

    processes = [0, 1]
    dataFrames = []
    minValues = []
    maxValues = []
    colors = ["k", "r"]
    processIndices = []

    for proc in processes:
        indexList = df[df["process"] == proc].index.tolist()
        processIndices.append((indexList[0], indexList[-1]))
        #print("indexList: {}".format(indexList))
        copiedDataFrame = df[indexList[0]: indexList[-1]].copy()
        dataFrames.append(copiedDataFrame.sort_values("key"))
        #dataFrames.append(copiedDataFrame.loc[indexList[0], indexList[-1]])

    print(processIndices)

    for dataFrame in dataFrames:
        print("min key: {} at {}".format(dataFrame["key"].min(), dataFrame["key"].idxmin()))
        minValues.append((dataFrame["key"].min(), dataFrame["key"].idxmin()))
        print("max key: {} at {}".format(dataFrame["key"].max(), dataFrame["key"].idxmax()))
        maxValues.append((dataFrame["key"].max(), dataFrame["key"].idxmax()))

    minParticles = []
    maxParticles = []

    for i in range(len(minValues)):
        #print(minValues[i])
        print("min(Particle) = ({}, {}, {})".format(df["x"][minValues[i][1]],
                                                    df["y"][minValues[i][1]],
                                                    df["z"][minValues[i][1]]))
        minParticles.append((df["x"][minValues[i][1]],
                            df["y"][minValues[i][1]],
                            df["z"][minValues[i][1]]))
        #print(maxValues[i])
        print("max(Particle) = ({}, {}, {})".format(df["x"][maxValues[i][1]],
                                                    df["y"][maxValues[i][1]],
                                                    df["z"][maxValues[i][1]]))
        maxParticles.append((df["x"][maxValues[i][1]],
                             df["y"][maxValues[i][1]],
                             df["z"][maxValues[i][1]]))

    for i in range(minValues[1][1], maxValues[1][1]):
        print("distance: {}  key: {}".format(round(distance(maxParticles[0], (df["x"][i], df["y"][i], df["z"][i])), 2),
                                                        df["key"][i]))

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #for i, processIndex in enumerate(processIndices):
    #    ax.scatter(df["x"][processIndex[0]:processIndex[1]],
    #               df["y"][processIndex[0]:processIndex[1]],
    #               df["z"][processIndex[0]:processIndex[1]],
    #               c=colors[i])
    #plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, dataFrame in enumerate(dataFrames):
        ax.scatter(dataFrame["x"],
                   dataFrame["y"],
                   dataFrame["z"],
                   c=colors[i])
        if connect:
            ax.plot(dataFrame["x"],
                    dataFrame["y"],
                    dataFrame["z"],
                    c=colors[i], alpha = 0.5)
    ax.set_axis_off()
    plt.show()
