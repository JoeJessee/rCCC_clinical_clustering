from ipykernel import kernelapp as app
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Read in clinical data and separate target labels from data
clinical_info = pd.read_csv("cleaned_clinical.csv", index_col=1)
clinical_info = clinical_info.drop(['Unnamed: 0'], axis=1)
clinical_info = clinical_info.dropna(subset=['tumor_stage'])
clinical_info_without_target = clinical_info.loc[:, clinical_info.columns !='tumor_stage']

# replace -- with Nan
clinical_info = clinical_info.replace('--', np.nan)
clinical_info = clinical_info.replace('not reported', np.nan)

# keep not reported in categorical variable, ethnicity and race
clinical_info['ethnicity'] = clinical_info['ethnicity'].replace(np.nan, 'not reported')
clinical_info['race'] = clinical_info['race'].replace(np.nan, 'not reported')

# convert columns with numbers to floats
to_float = ['year_of_birth','year_of_death', 'tumor_stage','age_at_diagnosis', 'days_to_death', 'days_to_last_follow_up']
for col in to_float:
    clinical_info[col] = clinical_info[col].astype('float64')

# add new column age_at_death
clinical_info['age_at_death'] = clinical_info['age_at_diagnosis'] + clinical_info['days_to_death']

# Read in RNA_Seq data
RNA_Seq = pd.read_csv("Primary_Tumor_FPKM.csv", index_col=0)

# Define object to hold all available patient data types
class whole_patient:
    def __init__(self, clinical, RNA_seq, patient_IDs, target, unit_cost=0):
        self.clinical = clinical
        self.RNA_seq = RNA_seq
        self.patient_IDs = patient_IDs
        self.target = target
        self.unit_cost = unit_cost

# Create object
whole_patient = whole_patient(clinical=clinical_info_without_target, RNA_seq=RNA_Seq, patient_IDs=clinical_info.index, target=clinical_info['tumor_stage'])

# Create function for getting all of the paired patient IDs and Tumor stages
def get_tumor_stages(self):
        dicts = {}
        keys = whole_patient.patient_IDs
        values = whole_patient.target
        for i in keys:
            dicts[i] = values[i]
        return(dicts)

# Encode factor data and replace the NaN's with 0
whole_patient.clinical = pd.get_dummies(whole_patient.clinical, dummy_na=True)
whole_patient.clinical = whole_patient.clinical.replace(np.nan, 0)

# Normalize Data, and then MinMax Scale it
from sklearn.preprocessing import Normalizer, MinMaxScaler
# using normalizer
norm = Normalizer()
norm.fit(whole_patient.clinical)
whole_patient.clinical = pd.DataFrame(norm.fit_transform(whole_patient.clinical), columns=whole_patient.clinical.columns, index=whole_patient.clinical.index)
# and then using MinMaxScaler
MMS = MinMaxScaler()
MMS.fit(whole_patient.clinical)
whole_patient.clinical = pd.DataFrame(MMS.fit_transform(whole_patient.clinical), columns=whole_patient.clinical.columns, index=whole_patient.clinical.index)

# Principle Component Analysis
from sklearn.decomposition import PCA
import seaborn as sb
pca = PCA(n_components = 9)
pca_of_clinical = pca.fit(whole_patient.clinical).transform(whole_patient.clinical)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

# Create Dataframe of Clinical PCA Values
clinical_PCA = pd.DataFrame()
clinical_PCA['PC1'] = pca_of_clinical[:,0]
clinical_PCA['PC2'] = pca_of_clinical[:,1]
clinical_PCA['PC3'] = pca_of_clinical[:,2]
clinical_PCA['PC4'] = pca_of_clinical[:,3]
clinical_PCA['PC5'] = pca_of_clinical[:,4]
clinical_PCA['PC6'] = pca_of_clinical[:,5]
clinical_PCA['PC7'] = pca_of_clinical[:,6]
clinical_PCA['PC8'] = pca_of_clinical[:,7]
clinical_PCA['PC9'] = pca_of_clinical[:,8]
clinical_PCA.index = whole_patient.clinical.index

# Plot Clinical PC1 vs. PC2
plt.figure(figsize=(10,7))
sb.scatterplot(x='PC1', y='PC2', data=clinical_PCA, hue=whole_patient.target, legend='brief');
plt.title('Clinical Data PC1 vs. PC2')
plt.legend(loc='upper left', prop={'size':8}, bbox_to_anchor=(1,1))
plt.tight_layout(pad=7)
plt.show()

# Elbow Plot of Principle Components
df = pd.DataFrame({'var':pca.explained_variance_ratio_,
             'PC':['PC1','PC2','PC3','PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9']})
plt.figure(figsize=(10,7))
sb.lineplot(x='PC',y="var",
           data=df, color="c")
plt.title('Elbow Plot')
plt.xlabel('Principle Components')
plt.ylabel('Percentage Variation Explained')
plt.show()

# Elbow is at PC4, don't need others
clinical_PCA = clinical_PCA.drop(['PC5', 'PC6', 'PC7', 'PC8', 'PC9'], axis = 1)

# T-SNE
# Create numpy array of each row in the dataset
tsne_list = []
for x in range(len(clinical_PCA.index)):
    row = clinical_PCA.iloc[x,:]
    tsne_list.append(row)
tsne_list = np.array(tsne_list)

# Reduce dimensionality with TSNE
from sklearn.manifold import TSNE
tsne_transformed = TSNE(n_components=3, verbose = 1, perplexity = 25, n_iter = 1000, learning_rate = 100, random_state=1).fit_transform(tsne_list)
# T-SNE was improved by toggling the perplexity, the learning rate, and n_iter

# Create a column for each TSNE component and add it to the dataframe
clinical_PCA['tsne-one'] = tsne_transformed[:,0]
clinical_PCA['tsne-two'] = tsne_transformed[:,1]
clinical_PCA['tsne-three'] = tsne_transformed[:,2]

# Plot using TSNE scatterplot
plt.figure(figsize=(10,7))
sb.scatterplot(x='tsne-one', y='tsne-two', data=clinical_PCA, hue=whole_patient.target)
plt.title('Clinical Data PC1 - PC4  (87% variation represented)')
plt.legend(loc='upper left', prop={'size':8}, bbox_to_anchor=(1,1))
plt.tight_layout(pad=7)
plt.show()

### MACHINE LEARNING ###

# Use a train_test_split and attempt to model the clinical data
from sklearn.model_selection import train_test_split
# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(whole_patient.clinical, whole_patient.target.values, test_size=.1)

# Instantiate KNN and fit to data
from sklearn.model_selection import StratifiedKFold, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# CV iterators
inner_cv_iterator = ShuffleSplit(n_splits=3, random_state=10)
outer_cv_iterator = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)

# (Hyper)parameter grid
p_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
}

# Instantiate KNN and wrap in GridSearchCV
knn = KNeighborsClassifier()
grid_search = GridSearchCV(estimator=knn, param_grid=p_grid, cv=inner_cv_iterator)

#Fit model to training set
grid_search.fit(X_train, y_train)
best = grid_search.best_params_
cv = grid_search.cv_results_

print('GridSearch found optimal number of neighbors:', best['n_neighbors'])
print('Mean CV test scores are:', cv['mean_test_score'])

knn = KNeighborsClassifier(n_neighbors = best['n_neighbors'])
knn.fit(X_train, y_train)

print('Accuracy:', accuracy_score(y_test, knn.predict(X_test)))

## Create confusion matrix for KNN
from sklearn.metrics import confusion_matrix
KNN_Y_predicted = knn.predict(X_train)
array = confusion_matrix(y_train, KNN_Y_predicted)
df_cm = pd.DataFrame(array, index = [i for i in ['True Stage 1', ' True Stage 2', 'True Stage 3', 'True Stage 4']],
                  columns = [i for i in ['Pred Stage 1', 'Pred Stage 2', 'Pred Stage 3', 'Pred Stage 4']])
plt.figure(figsize = (10,7))
sb.heatmap(df_cm, annot=True)

# Integration of RNA-Seq from here
