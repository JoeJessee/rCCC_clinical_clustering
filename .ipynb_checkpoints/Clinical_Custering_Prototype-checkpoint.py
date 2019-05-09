# Clustering of clinical info
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import seaborn as sns

# read in data
clinical_info = pd.read_csv("cleaned_clinical.csv")
# remove rows that don't have the label of interest
clinical_info = clinical_info.dropna(subset=['tumor_stage']);
clinical_info = clinical_info.drop(['Unnamed: 0'], axis=1)
clinical_info.columns
clinical_info = clinical_info.set_index('submitter_id')
clinical_info = clinical_info.rename_axis('submitter_id')
len(clinical_info.index)

# separate data from labels
clinical_info.target = np.array(clinical_info['tumor_stage'])
clinical_info.data = clinical_info.loc[:, clinical_info.columns != 'tumor_stage']

clinical_info.target
clinical_info.data
# use Pandas getdummies for encoding factor data
clinical_info.encoded = pd.get_dummies(clinical_info.data, dummy_na= True)

# test if there are any NAs in the df
clinical_info.encoded = clinical_info.encoded.replace(np.nan, 0)
clinical_info.encoded.isnull().values.any()

# assign variables for ML
y = clinical_info.target
X = clinical_info.encoded

# Principle component Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_of_cie = pca.fit(clinical_info.encoded).transform(clinical_info.encoded)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
print(pca.singular_values_)

clinical_info.data['pc-one'] = pca_of_cie[:,0]
clinical_info.data['pc-two'] = pca_of_cie[:,1]
sns.scatterplot(x='pc-one', y='pc-two', data=clinical_info.data, hue=clinical_info.target)


# T-SNE
# Create numpy array of each row in the dataset
tsne_list = []
len(clinical_info.encoded.index)
for x in range(len(clinical_info.encoded.index)):
    row = clinical_info.encoded.iloc[x,:]
    tsne_list.append(row)

tsne_list = np.array(tsne_list)

# Reduce dimensionality with TSNE
from sklearn.manifold import TSNE
tsne_transformed = TSNE(n_components=3, verbose = 1, perplexity = len(tsne_list), n_iter = 1000).fit_transform(tsne_list)

# Create a column for each TSNE component and add it to the dataframe
clinical_info.data['tsne-one'] = tsne_transformed[:,0]
clinical_info.data['tsne-two'] = tsne_transformed[:,1]
clinical_info.data['tsne-three'] = tsne_transformed[:,2]

# Plot using TSNE scatterplot
sns.scatterplot(x='tsne-one', y='tsne-two', data=clinical_info.data, hue=clinical_info.target)


# Instantiate KNN and fit to data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

pred_y = knn.predict(X)

type(pred_y)

# Compare the label with the predicted label for knn
comparison = []
def compare_arrays(y, pred_y):
    for x in range(len(y)):
        if y[x] == pred_y[x]:
            comparison.append(1)
        else:
            comparison.append(0)
    efficiency = (sum(comparison) / len(y)) * 100
    print("The label matches the predicted label " + str(efficiency) + " percent of the time.")

compare_arrays(y, pred_y)
