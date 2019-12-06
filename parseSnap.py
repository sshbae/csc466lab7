import sys
import time
import math
import numpy as np
import pandas as pd


def fillMatrix(matrixDf, inputDf):
    for index, row in inputDf.iterrows():
        print(index)
        matrixDf.loc[row[1], row[0]] = np.int8(1)

    nonzeros = matrixDf.astype(bool).sum(axis=0)
    matrixDf = matrixDf / nonzeros
    print("removing nan")
    matrixDf.fillna(np.int8(0), inplace=True)

    print("about to replace infs")
    matrixDf.replace(float("inf"), np.int8(1), inplace=True)
    return matrixDf

def getMatrix(labels, inputDf):
    temp = np.zeros((labels.size, labels.size), dtype = np.int8)
    matrixDf = pd.DataFrame(data=temp, index=labels, columns=labels)
    matrixDf = fillMatrix(matrixDf, inputDf)

    return matrixDf

# O(jk) is all outbound edges from jk
def pageRank(matrixDf, numNodes):
    print("working on ranking")
    d = 0.90
    e = 0.03
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

    return numIters, ranks

def report(readTime, procTime, numIters, matrixDf, ranks):
    with open("amazon.out", 'w') as f:
        f.write(f"Read time:       {readTime}\n")
        f.write(f"Iterations:      {numIters}\n")
        f.write(f"Processing time: {procTime}\n")
        f.write("    avg per node across all iterations\n")
        for i in range(len(ranks)):
            f.write(f"node: {matrixDf.index[i]} with pagerank: {ranks[i]}\n")

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

    report(readTime, procTime / numNodes, numIters, matrixDf, ranks)


if __name__ == '__main__':
    main()
