#using duplicates

import sys
import math
import pandas as pd
import numpy as np

def fillMatrix(df, inputDf):
    for index, row in inputDf.iterrows():
        val1 = row.at["node1ID"].strip(' "')
        val2 = row.at["node2ID"].strip(' "')

        #if node 1 is the winner
        if row.at["node1Val"] > row.at["node2Val"]:
            df.loc[val2,val1] = 1
            df.loc[val1,val2] = 0

        #if node 2 is the winner
        else:
            df.loc[val1,val2] = 1
            df.loc[val2,val1] = 0

    outCounts = df.sum(axis=0) #counts outdegrees for each node, represented by the index of this array
    #df = df.divide(outCounts)
    df = df/outCounts
    df = df.fillna(0)
    df = df.replace(float("inf"),1)
    return df

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
    infile = sys.argv[1]
    nodeNames = {}
    inputDf = pd.read_csv(infile, header=None, names= ["node1ID", "node1Val", "node2ID", "node2Val"])
    inputDf[["node1Val", "node2Val"]] = inputDf[["node1Val", "node2Val"]].apply(pd.to_numeric)

    uniqueVals = np.concatenate((pd.unique(inputDf["node1ID"]), pd.unique(inputDf["node2ID"])))
    uniqueVals = pd.unique(uniqueVals)
    for i in range(len(uniqueVals)):
        uniqueVals[i] = uniqueVals[i].strip(' "')

    filler = np.zeros((uniqueVals.size, uniqueVals.size))
    df = pd.DataFrame(filler, columns = uniqueVals, index = uniqueVals)
    matrixDf = fillMatrix(df, inputDf)

    numIters, ranks = pageRank(matrixDf, len(uniqueVals))

if __name__ == '__main__':
    main()
