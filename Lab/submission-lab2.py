## import modules here
import pandas as pd
import numpy as np
import helper


################### Question 1 ###################
def buc_rec_optimized(df):  # do not change the heading of the function
    columns = df.columns
    resultCuboid = pd.DataFrame(columns=columns)
    buc_rec_optimization(df, [], [], resultCuboid)
    return resultCuboid

def single_tuple_optimization(originalCuboid, originalRow, index, resultCuboid):
    if originalCuboid.shape[1] == 1:
        originalRow.append(originalCuboid.iloc[0, 0])
        resultCuboid.loc[len(index)] = originalRow
        originalRow.pop()
    else:
        originalRowTmp = originalRow.copy()
        base = list(originalCuboid.iloc[0, :])
        originalRowTmp.extend(base)
        resultCuboid.loc[len(index)] = originalRowTmp
        lastRow = []
        for i in range(len(base) - 1):
            lastRow.append('ALL')
        lastRow.append(originalCuboid.iloc[0, -1])

        rows = [base]
        row = []
        rowTmp = []
        originalRowTmp = originalRow.copy()
        i, j, k, l = 0, 0, 0, 0
        # index = 1
        index.append(1)
        while lastRow not in rows:
            while i < len(rows):
                length = len(rows)

                for k in range(len(rows[i])):  # deep copy
                    row.append(rows[i][k])

                for j in range(len(base) - 1):
                    if row[j] != 'ALL':

                        for l in range(len(row)):  # deep copy
                            rowTmp.append(row[l])

                        rowTmp[j] = 'ALL'
                        if rowTmp not in rows:
                            originalRowTmp.extend(rowTmp)
                            resultCuboid.loc[len(index)] = originalRowTmp
                            rows.append(rowTmp)
                            index.append(1)
                    rowTmp = []
                    originalRowTmp = originalRow.copy()
                row = []
                i += 1
        index.pop()


def buc_rec_optimization(originalCuboid, row, index, resultCuboid):
    if originalCuboid.shape[0] == 1:
        single_tuple_optimization(originalCuboid, row, index, resultCuboid)

    elif originalCuboid.shape[1] == 1:
        sumOfMeasureValues = sum(helper.project_data(originalCuboid, 0))
        row.append(sumOfMeasureValues)
        resultCuboid.loc[len(index)] = row
        row.pop()
    else:
        firstDimValus = sorted(list(set(helper.project_data(originalCuboid, 0).values)))
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

