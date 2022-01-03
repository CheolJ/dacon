import pandas as pd

project_name = 'penguin_weight_prediction'

# read files
tr_raw = pd.read_csv(project_name+'/dataset/train.csv')
test = pd.read_csv(project_name+'/dataset/test.csv')
result = pd.read_csv(project_name+'/dataset/sample_submission.csv')

# null-check
print("Check Null data in train dataset\n",tr_raw.isnull().sum())
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
print("Check Null data in test dataset\n",test.isnull().sum())
print("*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*")
tr_ = tr_raw.copy()

# one-hot-encoding
print("Process : One hot encoding")
tr_.replace({'Yes': 1, 'No': 0, 'MALE': 1, "FEMALE": 0}, inplace=True)
test.replace({'Yes': 1, 'No': 0, 'MALE': 1, "FEMALE": 0}, inplace=True)
tr_ = pd.concat([tr_, pd.get_dummies(tr_[['Island','Species']])], axis=1)
test = pd.concat([test, pd.get_dummies(test[['Island','Species']])], axis=1)
tr_.drop(['Island', 'Species'], axis=1, inplace=True)
test.drop(['Island', 'Species'], axis=1, inplace=True)
print("Complete : One hot encoding")
print(tr_.columns)
print(test.columns)

#