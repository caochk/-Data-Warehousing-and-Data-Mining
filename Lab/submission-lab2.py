## import modules here 
import pandas as pd
import numpy as np
import helper


################### Question 1 ###################
def read_data(filename):
    baseCuboid = pd.read_csv(filename, sep='\t')
    return (baseCuboid)


def buc_rec_optimized(df):# do not change the heading of the function
    # baseCuboid = df
    if df.shape[0] == 1:
        # print(df)
        single_tuple_optimization(df)
    else:
        buc_rec_optimization()


def single_tuple_optimization(baseCuboid):
    # print(baseCuboid)
    base = list(baseCuboid.iloc[0, 0:-1])
    # print(base)
    columns = list(baseCuboid.iloc[:, 0:-1])
    # print(columns)
    resultCuboid = pd.DataFrame(columns=columns)
    # print(resultCuboid)
    resultCuboid.loc['0'] = base
    # print(resultCuboid)
    lastRow = []
    for i in range(len(base)):
        lastRow.append('ALL')
    # print(lastRow)

    rows = [base]
    row = []
    rowTmp = []
    i, j, k, l = 0, 0, 0, 0
    index = 1
    while lastRow not in rows:
        while i < len(rows):
            length = len(rows)

            for k in range(len(rows[i])): #深拷贝过程
                row.append(rows[i][k])

            for j in range(len(base)):
                if row[j] != 'ALL':

                    for l in range(len(row)): #深拷贝过程
                        rowTmp.append(row[l])

                    rowTmp[j] = 'ALL'
                    if rowTmp not in rows:
                        resultCuboid.loc[str(index)] = rowTmp
                        rows.append(rowTmp)
                        index += 1
                rowTmp = []
            row = []
            # print(i, ",")
            i += 1

    # print(rows)
    # print(resultCuboid)

    resultCuboid[baseCuboid.columns[-1]] = baseCuboid.iloc[0, -1]
    print(resultCuboid)

def buc_rec_optimization(cuboid):
    pass


baseCuboid = read_data('c_.txt')
buc_rec_optimized(baseCuboid)