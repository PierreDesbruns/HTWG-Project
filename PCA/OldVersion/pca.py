import numpy as np
import pandas as pd
import statistics as stat
import matplotlib.pyplot as plt
from math import sqrt


"""
Variable to use from this program:
    * anomaly_matrix
        array containing 0 or 1 (1 if related value is an anomaly; 0 if not); has the same size than the data array

/!\ All data arrays used here are transposed compared to csv dataframe
"""


# ================================================================================================================
# FUNCTIONS
# ================================================================================================================

def rowMean(array):
    means = []
    for row in range(0, len(array)):
        means.append(stat.mean(array[row,:]))
    return means

def rowVariance(array):
    variances = []
    for row in range(0, len(array)):
        variances.append(stat.variance(array[row,:]))
    return variances


# ================================================================================================================
# MAIN
# ================================================================================================================

file = open("Datas_with_dots.csv", "r")

# 0. Initialization
datas_csv = pd.read_csv(file, usecols = [0,1,2,3,4,5,6,7,8,9,10,11])
datas_csv_columns = datas_csv.columns
datas_transposed = datas_csv.to_numpy()
datas = datas_transposed.T
nbRows = len(datas)
nbColumns = len(datas[0])
datas_means = rowMean(datas)
datas_variances = rowVariance(datas)

# 1. Standardization of initial datas
datas_standardized = np.zeros([nbRows, nbColumns], dtype=float)
for row in range(0, nbRows):
    for column in range(0, nbColumns):
        datas_standardized[row, column] = (datas[row, column] - datas_means[row]) / sqrt(datas_variances[row])

# 2. Diagonalization of the covariances matrix
cov_matrix = np.cov(datas_standardized, bias = False)
cov_eigVal, cov_eigVec = np.linalg.eig(cov_matrix)
cov_matrix_trace = np.sum(cov_eigVal)
cov_diagMatrix = np.diag(cov_eigVal)
cov_changeBasis = cov_eigVec
cov_changeBasis_inv = np.linalg.inv(cov_changeBasis)

# 3. Eigenvalues selection
cov_eigVal_threshold = 0.1                              #/!\ ARBITRARY
cov_eigVal_modified = []
for eigValIndex in range(0, len(cov_eigVal)):
    if cov_eigVal[eigValIndex] / cov_matrix_trace > cov_eigVal_threshold:
        cov_eigVal_modified.append(cov_eigVal[eigValIndex])
    else:
        cov_eigVal_modified.append(0)

# 4. Projection (from initial coordinate system to principal components coordinate system)
pca_datas_transformed = np.zeros([nbRows, nbColumns], dtype=float)
for row in range(0, nbRows):
    for column in range(0, nbColumns):
        pca_datas_transformed[row, column] = (np.dot(cov_changeBasis[:,row], datas_standardized[:,column])) * cov_eigVal_modified[row] / cov_matrix_trace

# 5. Inverse projection (from principal components coordinate system to initial coordinate system)
pca_datas_standardized = np.zeros([nbRows, nbColumns], dtype=float)
for row in range(0, nbRows):
    for column in range(0, nbColumns):
        pca_datas_standardized[row, column] = np.dot(cov_changeBasis_inv[:,row], pca_datas_transformed[:,column])

# 6. Inverse standadization of modified datas
pca_datas = np.zeros([nbRows, nbColumns], dtype=float)
for row in range(0, nbRows):
    for column in range(0, nbColumns):
        pca_datas[row, column] = pca_datas_standardized[row, column] * sqrt(datas_variances[row]) + datas_means[row]

# 7. Error gestion
error_matrix = np.zeros([nbRows, nbColumns], dtype=float)
for row in range(0, nbRows):
    for column in range(0, nbColumns):
        error_matrix[row, column] = (pca_datas[row, column] - datas[row, column])**2
error_means = rowMean(error_matrix)
error_variances = rowVariance(error_matrix)

# 8. Anomaly gestion
anomaly_matrix = np.zeros([nbRows, nbColumns], dtype=int)
anomaly_nbStandDev = 2                                    #/!\ ARBITRARY
for row in range(0, nbRows):
    for column in range(0, nbColumns):
        if abs(error_matrix[row, column] - error_means[row]) < anomaly_nbStandDev * sqrt(error_variances[row]):
            anomaly_matrix[row, column] = 0
        else:
            anomaly_matrix[row, column] = 1


# ================================================================================================================
# PRINT
# ================================================================================================================

print("Eignevalues :")
print(cov_eigVal)
print("Modified eigenvalues :")
print(cov_eigVal_modified)
for row in range(0, nbRows):
    print(f"Column #{row} countains {np.sum(anomaly_matrix[row,:])} anomalies")


# ================================================================================================================
# PLOT
# ================================================================================================================

timestamp = np.array(range(0, nbColumns))

for row in range(0, nbRows):
    plt.figure(row)
    plt.title(f"{datas_csv_columns[row]}")
    plt.xlabel("Timestamp")
    plt.ylabel(f"Column #{row}")
    plt.plot(timestamp, datas[row,:])
    for column in range(0, nbColumns):
        if anomaly_matrix[row, column]:
            plt.axvline(timestamp[column], color='r', alpha=0.1)
    # plt.savefig(f"column{row}_threshold{cov_eigVal_threshold}.png")
plt.show()