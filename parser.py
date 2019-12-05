#using duplicates

import sys
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
    #df = df.divide(outCounts, axis=0)
    df = df/outCounts
    df = df.fillna(0)
    df = df.replace(float("inf"),1)

    return df

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

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(matrixDf)

if __name__ == '__main__':
    main()
