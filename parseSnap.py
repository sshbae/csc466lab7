import sys
import time
import math
import numpy as np
import pandas as pd


def fillMatrix(matrixDf, inputDf):
    for index, row in inputDf.iterrows():
        matrixDf.loc[row[1], row[0]] = 1

    nonzeros = matrixDf.astype(bool).sum(axis=0)
    matrixDf = matrixDf / nonzeros
    matrixDf = matrixDf.fillna(0)

    return matrixDf.replace(float("inf"),1)

def getMatrix(labels, inputDf):
    temp = np.zeros((labels.size, labels.size))
    matrixDf = pd.DataFrame(data=temp, index=labels, columns=labels)
    matrixDf = fillMatrix(matrixDf, inputDf)
   
    return matrixDf

# O(jk) is all outbound edges from jk
def pageRank(matrixDf, numNodes):
    d = 0.85
    e = 0.01
    prefix = (1 - d) * 1/numNodes
    ranks = []
    prevRanks = np.full(numNodes, 1/numNodes)
    diffs = 1

    numIters = 0
    while abs(diffs) > e:
        temp = matrixDf.mul(prevRanks)
        ranks = ((np.array(temp.sum(axis=1))) * d) + prefix
        diffs = np.sum(ranks - prevRanks)

        prevRanks = ranks
        numIters += 1
        print(diffs)

    return numIters, ranks

def main():
    start = time.time()
    inputDf = pd.read_csv(sys.argv[1], skiprows=[0, 1, 2, 3], 
        delim_whitespace=True, header=None, names=["From", "To"])

    labels = np.concatenate((pd.unique(inputDf["From"]), pd.unique(inputDf["To"])))
    labels = np.unique(labels)
    numNodes = len(labels)

    matrixDf = getMatrix(labels, inputDf)
    end = time.time()
    readTime = end - start

    start = time.time()
    numIters, ranks = pageRank(matrixDf, numNodes)
    end = time.time()
    procTime = end - start



if __name__ == '__main__':
    main()