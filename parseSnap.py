import sys;
import time;
import numpy as np;
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
    prefix = (1 - d) * 1/numNodes
    prevRanks = np.full(numNodes, 1/numNodes)
    temp = matrixDf.mul(prevRanks)
    ranks = ((np.array(temp.sum(axis=1))) * d) + prefix
    print(ranks)

    return 1, ranks

def main():
    procTime = 0

    start = time.time()
    inputDf = pd.read_csv(sys.argv[1], skiprows=[0, 1, 2, 3], 
        delim_whitespace=True, header=None, names=["From", "To"])

    labels = np.concatenate((pd.unique(inputDf["From"]), pd.unique(inputDf["To"])))
    labels = np.unique(labels)
    numNodes = len(labels)

    matrixDf = getMatrix(labels, inputDf)
    end = time.time()
    readTime = end - start

    numIters, ranks = pageRank(matrixDf, numNodes)



if __name__ == '__main__':
    main()