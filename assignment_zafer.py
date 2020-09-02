import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#loading the dataset
x= pd.read_csv('PCA_practice_dataset.csv',header=None)




#standardizing the features of the dataset
sc = StandardScaler()
df_std = sc.fit_transform(x)



#constructing the covariance matrix. The covariance matrix stores the pairwise covariances between the different features
cov_mat = np.cov(df_std.T)

#computing the eigenvectors and eigenvalues of the covariance matrix
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Making a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]

# Sorting the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)


for step in range(8):
  print('CASE : ',step+1)

  # Calculating the thereshold from the step number (initial thereshold is 0.9 and the step size is 0.01)
  thereshold = 0.9+(step*0.01)

  print('Thereshold for this case = ',thereshold)
  pcVec = []

  sum_total_eigen_values = np.sum([i[0] for i in eigen_pairs])
  sum_sel_eigen_values = 0

  # From the sorted list of eigenpairs, we pick the eigenvectors in order such that the sum of the variance explained ratio of an eigenvalues of the selected eigenvectors is just less than (or equal to) the thereshold.
  # The variance explained ratio of an eigenvalue is simply the fraction of an eigenvalue and the total sum of the eigenvalues
  for i in range(len(eigen_pairs)):
    sum_sel_eigen_values = sum_sel_eigen_values + eigen_pairs[i][0]
    if((sum_sel_eigen_values / sum_total_eigen_values)<= thereshold):
      pcVec.append(eigen_pairs[i][1][:, np.newaxis])
    else:
      break

  # We stack the sequence of selected eigenvectors horizontally (i.e. column wise) to make the projection matrix
  w=tuple(i for i in pcVec)
  w=np.hstack(w)

  # We use the projection matrix to transform the data onto the lower-dimensional subspace
  df_pca = df_std.dot(w)

  sum_pc=np.sum([eigen_pairs[i][0] for i in range(len(pcVec))])
  pc=[(eigen_pairs[i][0]/sum_pc) for i in range(len(pcVec))]

  # Now we plot the scree plot for this specific thereshold
  print('The Scree plot for this case is:')

  plt.plot(['PC'+str(i) for i in range(1,len(pc)+1)],pc,'o-')
  plt.ylabel('Explained variance ratio')
  plt.xlabel('Principal component index')
  plt.show()

  # Now we show the number of principal components retained due to the thresholding by simply counting the number of selected eigenvectors to make the projection matrix
  print('The number of principal components retained due to the thresholding = ',len(pc))