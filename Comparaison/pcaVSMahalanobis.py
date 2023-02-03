from csv import QUOTE_NONE
import numpy as np
import pandas as pd
import statistics as stat
from math import sqrt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.linear_model import LinearRegression
from scipy import stats
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance

# ================================================================================================================
# FUNCTIONS
# ================================================================================================================
def get_anomaly_scores(df_original, df_restored):
    loss = np.sum((np.array(df_original) - np.array(df_restored)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=df_original.index)
    return loss

def is_anomaly(data, pca, threshold):
    pca_data = pca.transform(data)
    restored_data = pca.inverse_transform(pca_data)
    loss = np.sum((data - restored_data) ** 2)
    print(loss)
    return loss >= threshold

def mahalanobis(x):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    """
    cov = np.cov(x, rowvar=False)
    if(cov.size == 1):
        inv_covmat = 1 / cov
    else:
        inv_covmat = np.linalg.inv(cov)
    mean=np.mean(x, axis=0)
    distance=np.zeros(len(x))
    for i, data in enumerate(x):
        if(cov.size == 1):
            distance[i]=(x[i]-mean)*(inv_covmat)*(x[i] - mean)
        else:
            distance[i]=(x[i]-mean).T.dot(inv_covmat).dot(x[i] - mean)
    return distance

def dateParser(date):
    date = date.split('.')[0]
    return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")

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

risk = 5/100

"""
===================================================================================================================
PCA PIERRE
===================================================================================================================
"""
file = open("../Datas_with_dots.csv", "r")

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
cov_eigVal_modified = []
cov_eigVal_threshold = 0.99                            #/!\ ARBITRARY
cov_selectedEigVal_sum = 0
cov_selectedEigVal_index = 0
while(cov_selectedEigVal_sum / cov_matrix_trace < cov_eigVal_threshold):
    cov_eigVal_modified.append(cov_eigVal[cov_selectedEigVal_index])
    cov_selectedEigVal_sum += cov_eigVal[cov_selectedEigVal_index]
    cov_selectedEigVal_index += 1
for eigValIndex in range(cov_selectedEigVal_index, len(cov_eigVal)):
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
file.close()
"""
===================================================================================================================
PCA NOLANE
===================================================================================================================
"""
f = open("../Datas_with_dots.csv", "r")

dataset = pd.read_csv(f, date_parser=dateParser, keep_default_na=False, quoting=QUOTE_NONE, engine="python")
date = dataset["timestamp"].to_numpy()
dataset = dataset.drop(dataset.columns[[dataset.shape[1]-2,dataset.shape[1]-1]], axis=1)

average_summer = "19.2" #average temp in summer in germany
threshold = 2
threshold_components = 0.95
#https://github.com/andrewm4894/colabs/blob/master/time_series_anomaly_detection_with_pca.ipynb

season_find = "DSP_KK_CIRCUIT__24A1_AUSSENTEMPERATUR_EQUIP_ID_Value"
week=1
array_season = dataset[season_find].to_numpy()
begin_summer = ""
summer=False
for i in range(0,len(array_season), 96*7): 
    avg = np.mean(array_season[i:i+96*7])# mean of a week 
    week+=1
    if(avg > 19.6 and summer==False): #  we are in summer if the average temp in a week is more than the threshold
        begin_summer=i
        summer=True

#split in two the arrays
dataset_winter = dataset.iloc[:begin_summer]
dataset_summer = dataset.iloc[begin_summer:]
merge = []
label_ct = 0

for data_season in (dataset_winter.to_numpy(), dataset_summer.to_numpy()):
    #scaling the data : centering it and reducing
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_season)
    scaled_df = pd.DataFrame(scaled_data)

    pca = PCA(random_state=0)

    pca.fit_transform(scaled_data)
    #doing the pca 

    sum = 0
    for ct, i in enumerate(pca.explained_variance_ratio_) :
        sum += i
        if(sum > threshold_components): # find how much components we want to keep
            break

    pca.set_params(n_components=ct) #keeping n components 
    pca.fit_transform(scaled_data) #must reload the data I know

    df_pca = pd.DataFrame(pca.fit_transform(scaled_data)) #put pca in a dataframe
    df_restored = pd.DataFrame(pca.inverse_transform(df_pca)) # restoring the data
    df_restored = scaler.inverse_transform(df_restored) # de-scaling it 
    df_restored = pd.DataFrame(df_restored, columns=dataset.columns)

    merge.append(df_restored)
    label_ct += 1

df_restored = pd.concat(merge, ignore_index=True) #merging the two arrays
array = dataset.to_numpy()

num_errors_pierre = []
num_errors_nolane = []
num_errors_mahalanobis = []
num_same_errors_pierre_nolane = []
num_same_errors_nolane_mahalanobis = []
num_same_errors_pierre_mahalanobis = []
num_same_errors_all = []

for i in range(0, len(array[0])-1): # for each column
    plt.figure(i) # for the plot (new figure each time)
    plt.title(dataset.columns[i])
    plt.plot(array[:,i], color='k')

    num_errors_pierre.append(0)
    num_errors_nolane.append(0)
    num_errors_mahalanobis.append(0)
    num_same_errors_pierre_nolane.append(0)
    num_same_errors_nolane_mahalanobis.append(0)
    num_same_errors_pierre_mahalanobis.append(0)
    num_same_errors_all.append(0)

    #threshold for winter / summer : mean +/- 2 sigma
    threshold_error_winter = (df_restored.to_numpy()[:,i] - array[:,i])**2
    threshold_i_winter = np.mean(threshold_error_winter) + threshold*np.std(threshold_error_winter)
    threshold_i_2_winter = np.mean(threshold_error_winter) - threshold*np.std(threshold_error_winter)
    threshold_error_summer = (df_restored.to_numpy()[:,i] - array[:,i])**2
    threshold_i_summer = np.mean(threshold_error_summer) + threshold*np.std(threshold_error_summer)
    threshold_i_2_summer = np.mean(threshold_error_summer) - threshold*np.std(threshold_error_summer)

    res = mahalanobis(dataset.to_numpy()[:,i])
    level = stats.chi2.ppf(1-risk, df=1)

    for j in range(0, len(array)):
        if(i < begin_summer ): #winter
            if(threshold_error_winter[j] > threshold_i_winter or threshold_error_winter[j] < threshold_i_2_winter): # it is an error
                num_errors_nolane[i] += 1
                if anomaly_matrix[i,j]:     
                    num_errors_pierre[i] += 1
                    num_same_errors_pierre_nolane[i] += 1
                    if(abs(res[j]) > level):#PCA NOLANE & PCA PIERRE & Mahalanobis : rouge
                        num_errors_mahalanobis[i] += 1
                        num_same_errors_nolane_mahalanobis[i] += 1
                        num_same_errors_pierre_mahalanobis[i] += 1
                        num_same_errors_all[i] += 1
                        plt.axvline(x=j, color='r', alpha=0.1)
                    else:                   #PCA NOLANE & PIERRE
                        plt.axvline(x=j, color='tab:brown', alpha=0.1)
                elif(abs(res[j]) > level):  #PCA NOLANE & Mahalanobis : bleu
                    num_errors_mahalanobis[i] += 1
                    num_same_errors_nolane_mahalanobis[i] += 1
                    plt.axvline(x=j, color='b', alpha=0.1)
                else:                       #PCA NOLANE : jaune
                    plt.axvline(j, color='y', alpha=0.1)
            else:
                if anomaly_matrix[i,j]:     #PCA PIERRE & Mahalanobis : magenta
                    num_errors_pierre[i] += 1
                    if(abs(res[j]) > level):
                        num_errors_mahalanobis[i] += 1
                        num_same_errors_pierre_mahalanobis[i] += 1
                        plt.axvline(x=j, color='m', alpha=0.1)
                    else:                   #PCA PIERRE : cyan
                        plt.axvline(x=j, color='c', alpha=0.1)
                else:
                    if(abs(res[j]) > level):    #Mahalanobis : vert
                        num_errors_mahalanobis[i] += 1
                        plt.axvline(x=j, color='g', alpha=0.1)
        else: # summer
            if(threshold_error_summer[j] > threshold_i_summer or threshold_error_summer[j] < threshold_i_2_summer):
                num_errors_nolane += 1
                if anomaly_matrix[i,j]:
                    num_errors_pierre += 1
                    if(abs(res[j]) > level):#PCA NOLANE & PCA PIERRE & Mahalanobis : rouge
                        num_errors_mahalanobis[i] += 1
                        num_same_errors_all[i] += 1
                        plt.axvline(x=j, color='r', alpha=0.1)
                    else:                   #PCA NOLANE & PIERRE
                        plt.axvline(x=j, color='tab:brown', alpha=0.1)
                elif(abs(res[j]) > level):  #PCA NOLANE & Mahalanobis : bleu
                    num_errors_mahalanobis[i] += 1
                    num_same_errors_nolane_mahalanobis[i] += 1
                    plt.axvline(x=j, color='b', alpha=0.1)
                else:                       #PCA NOLANE : jaune
                    plt.axvline(j, color='y', alpha=0.1)
            else:
                if anomaly_matrix[i,j]:     #PCA PIERRE & Mahalanobis : magenta
                    num_errors_pierre[i] += 1
                    if(abs(res[j]) > level):
                        num_errors_mahalanobis[i] += 1
                        num_same_errors_pierre_mahalanobis[i] += 1
                        plt.axvline(x=j, color='m', alpha=0.1)
                    else:                   #PCA PIERRE : cyan
                        plt.axvline(x=j, color='c', alpha=0.1)
                else:
                    if(abs(res[j]) > level):    #Mahalanobis : vert
                        num_errors_mahalanobis[i] += 1
                        plt.axvline(x=j, color='g', alpha=0.1)

    print(dataset.columns[i])
    print("Errors PCA NOLANE :", num_errors_nolane[i]) 
    print("Errors PCA PIERRE :", num_errors_pierre[i]) 
    print("Errors Mahalanobis :", num_errors_mahalanobis[i])
    print("Errors PCA NOLANE & Mahalanobis :", num_same_errors_nolane_mahalanobis[i]) 
    print("Errors PCA PIERRE & Mahalanobis :", num_same_errors_pierre_mahalanobis[i]) 
    print("Errors PCA NOLANE & PIERRE :", num_same_errors_pierre_nolane[i])
    print("Errors in common :", num_same_errors_all[i])

    #plt.savefig(f"column{i}_pcaThreshold{cov_eigVal_threshold}_risk{risk}.png")

print("Errors PCA NOLANE (mean) :", stat.mean(num_errors_nolane)) 
print("Errors PCA PIERRE (mean) :", stat.mean(num_errors_pierre)) 
print("Errors Mahalanobis (mean) :", stat.mean(num_errors_mahalanobis))
print("Errors PCA NOLANE & Mahalanobis (mean) :", stat.mean(num_same_errors_nolane_mahalanobis)) 
print("Errors PCA PIERRE & Mahalanobis (mean) :", stat.mean(num_same_errors_pierre_mahalanobis)) 
print("Errors PCA NOLANE & PIERRE (mean) :", stat.mean(num_same_errors_pierre_nolane)) 
print("Errors in common (mean) :", stat.mean(num_same_errors_all))

plt.show()