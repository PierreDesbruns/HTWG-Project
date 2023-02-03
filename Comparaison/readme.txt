pcaVSMahalanobis.py script:
     > Compares two PCAs (from Nolane and from Pierre) to the Mahalanobis distance analysis
     > Parameters considered
          * cov_eigval_threshold (threshold of the new version of Pierre's PCA)
          * risk (risk taken of picking an error in Mahalanobis distance analysis)
          * /!\ No parameter of Nolane's PCA is considered

Pictures file:
     > Contains pictures and results of different values of considered parameters
     > Captions:
          * black: datas of each column
          * yellow: anomalies detected by Nolane's PCA
          * cyan: anomalies detected by Pierre's PCA
          * green: anomalies detected by Mahalanobis distance
          * blue: anomalies detected by both Nolane's PCA and Mahalanobis distance
          * magenta: anomalies detected by both Pierre's PCA and Mahalanobis distance
          * brown: anomalies detected by both Nolane's and Pierre's PCA
          * red: anomalies detected by all of three analysis