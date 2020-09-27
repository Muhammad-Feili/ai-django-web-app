import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# load dataset
dataset = pd.read_csv('data/train.csv', encoding='latin-1')
dataset = dataset.rename(columns=lambda x : x.strip().lower())
dataset.head()

# cleaning missing values
dataset = dataset[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]
dataset['sex'] = dataset['sex'].map({"male":0, "female":1})
dataset['age'] = pd.to_numeric(dataset['age'], errors='coerce')
dataset['age'] = dataset['age'].fillna(np.mean(dataset['age']))

# dummy variables
embarked_dummies = pd.get_dummies(dataset['embarked'])
dataset = pd.concat([dataset, embarked_dummies], axis=1)
dataset = dataset.drop(['embarked'], axis=1)

X = dataset.drop(['survived'], axis=1)
Y = dataset['survived']

# scaling features
sc = MinMaxScaler(feature_range=(0,1))
X_scaled = sc.fit_transform(X)

# model fit
log_model = LogisticRegression(C=1)
log_model.fit(X_scaled, Y)

# saving model as a pickle
import pickle

model_outfile = open("titanic_survival_ml_model.sav", "wb")
scaler_outfile = open("scaler.sav", "wb")

pickle.dump(log_model, model_outfile)
pickle.dump(sc, scaler_outfile)

model_outfile.close()
scaler_outfile.close()
