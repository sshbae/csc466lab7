#CSC466 F19 Lab 7
#Sarah Bae, shbae@calpoly.edu
#Roxanne Miller, rmille60@calpoly.edu
#smallDatasets.py <csv file name>

import sys
import time
import math
import pandas as pd
import numpy as np

def fillMatrix(df, inputDf):
    pd.set_option('display.max_rows', None)
    for index, row in inputDf.iterrows():
        val1 = str(row.at["node1ID"]).strip(' "')
        val2 = str(row.at["node2ID"]).strip(' "')

        #if node 1 is the winner
        if row.at["node1Val"] > row.at["node2Val"]:
            df.loc[val2,val1] = 1

        #if node 2 is the winner
        else:
            df.loc[val1,val2] = 1

    outCounts = df.sum(axis=0) #counts outdegrees for each node, represented by the index of this array
    df = df/outCounts
    df = df.fillna(0)
    df = df.replace(float("inf"),1)
    return df

# O(jk) is all outbound edges from jk
def pageRank(matrixDf, numNodes):
    d = 0.9
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

    return numIters, ranks
 
def displayRanks(ranks, uniqueVals):
    tupList = np.array([ranks, uniqueVals]).T
    np.set_printoptions(threshold=sys.maxsize)
    tupList = sorted(tupList, key=lambda tup: tup[0], reverse=True)
    for i in range(len(tupList)):
        
        print(f"{i+1} obj: {tupList[i][1]} with pagerank: {tupList[i][0]}") 

def main():
    start = time.time()
    nodeNames = {}
    inputDf = pd.read_csv(sys.argv[1], header=None, names= ["node1ID", "node1Val", "node2ID", "node2Val"])
    inputDf[["node1Val", "node2Val"]] = inputDf[["node1Val", "node2Val"]].apply(pd.to_numeric)

    uniqueVals = np.concatenate((pd.unique(inputDf["node1ID"]), pd.unique(inputDf["node2ID"])))
    for i in range(len(uniqueVals)):
        uniqueVals[i] = str(uniqueVals[i]).strip(' "')
    uniqueVals = pd.unique(uniqueVals)

    uniqueVals = uniqueVals.astype(str)
    filler = np.zeros((uniqueVals.size, uniqueVals.size))
    df = pd.DataFrame(filler, columns = uniqueVals, index = uniqueVals)
    matrixDf = fillMatrix(df, inputDf)
    end = time.time()
    readTime = end - start

    start = time.time()
    numIters, ranks = pageRank(matrixDf, len(uniqueVals))
    end = time.time()
    procTime = end - start

    displayRanks(ranks, uniqueVals)
    print(f"\nRead time: {readTime} sec\n"
            f"Processing time: {procTime/(numIters * len(uniqueVals))} sec avg per node across all iterations\n"
            f"Iterations: {numIters}")

if __name__ == '__main__':
    main()
