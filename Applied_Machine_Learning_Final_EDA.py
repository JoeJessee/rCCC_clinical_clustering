# Applied Machine Learning Final Project EDA
import numpy as np
import pandas as pd

# Read in data
clinical_info = pd.read_csv("cleaned_clinical.csv")

# Change null factors to np.nan
clinical_info.replace('not reported', np.nan);
clinical_info.replace('--', np.nan);

# Convert clinical
clinical_info = clinical_info.replace('stage i', 1)
clinical_info = clinical_info.replace('stage ii', 2)
clinical_info = clinical_info.replace('stage iii', 3)
clinical_info = clinical_info.replace('stage iv', 4)
clinical_info = clinical_info.replace('not reported', np.nan)


# view head of clinical data
clinical_info.head();
clinical_info.tumor_stage.mean()
