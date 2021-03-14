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
        row = []
        index = []
        columns = list(baseCuboid.iloc[:, :])
        resultCuboid = pd.DataFrame(columns=columns)
        buc_rec_optimization(df, row, index, resultCuboid)
        # print(resultCuboid)


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
    # print(len(baseCuboid.columns))

    resultCuboid[baseCuboid.columns[-1]] = baseCuboid.iloc[0, -1]
    # print(resultCuboid)

def base_case(originalCuboid, row, index, resultCuboid):
    sumOfMeasureValues = originalCuboid.iloc[:, 0].sum()
    row.append(sumOfMeasureValues)
    resultCuboid.loc[str(len(index))] = row

def buc_rec_optimization(originalCuboid, row, index, resultCuboid):
    if len(originalCuboid.columns) == 1:
        base_case(originalCuboid, row, index, resultCuboid)
        row.pop()
    else:
        firstDimValus = set(originalCuboid.iloc[:, 0])

        for value in firstDimValus:
            row.append(value)
            subCuboid = helper.slice_data_dim0(originalCuboid, value)
            buc_rec_optimization(subCuboid, row, index, resultCuboid)
            row.pop()
            index.append(1)

        row.append("ALL")
        subCuboid = helper.remove_first_dim(originalCuboid)
        buc_rec_optimization(subCuboid, row, index, resultCuboid)
        row.pop()
        # index.append(1)


baseCuboid = read_data('a_.txt')
buc_rec_optimized(baseCuboid)