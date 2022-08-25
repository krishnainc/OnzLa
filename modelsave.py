import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib

# Get the data
df = pd.read_csv('Travel_Ins.csv')

# Rename the column to remove spaces
df.rename(columns={'Agency Type': 'Type', 'Distribution Channel': 'Channel', 'Product Name': 'Product',
          'Net Sales': 'Net_Sales', 'Commision (in value)': 'Commision'}, inplace=True)

# separate the column
# Separate the column into numerical and categorical
numerical = ['Duration', 'Net_Sales', 'Commision', 'Age']
categorical = ['Agency', 'Type', 'Channel',
               'Product', 'Claim', 'Destination', 'Gender']

df = df[numerical+categorical]

df = df.drop(['Gender'], axis=1)

list_1 = df[(df['Age'] == 118)].index
df.drop(list_1, inplace=True)
list_2 = df[(df['Duration'] > 4000)].index
df.drop(list_2, inplace=True)
list_3 = df[(df['Duration'] == 0)].index
df.drop(list_3, inplace=True)


# encoding
label_encoder1 = LabelEncoder()
df['Agency'] = label_encoder1.fit_transform(df['Agency'])

label_encoder2 = LabelEncoder()
df['Type'] = label_encoder2.fit_transform(df['Type'])

label_encoder3 = LabelEncoder()
df['Channel'] = label_encoder3.fit_transform(df['Channel'])

label_encoder4 = LabelEncoder()
df['Product'] = label_encoder4.fit_transform(df['Product'])

label_encoder5 = LabelEncoder()
df['Claim'] = label_encoder5.fit_transform(df['Claim'])

label_encoder6 = LabelEncoder()
df['Destination'] = label_encoder6.fit_transform(df['Destination'])

column_names = ["Agency", "Type", "Channel", "Product", "Duration",
                "Destination", "Net_Sales", "Commision", "Age", "Claim"]
df = df.reindex(columns=column_names)

# feature selection
df = df.drop(['Channel'], axis=1)
df = df.drop(['Type'], axis=1)
df = df.drop(['Net_Sales'], axis=1)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


#oversampling + undersampling
smenn = SMOTEENN(random_state=42)
X_smenn, y_smenn = smenn.fit_resample(X, y)

X_train_smenn, X_test_smenn, y_train_smenn, y_test_smenn = train_test_split(
    X_smenn, y_smenn, test_size=0.2, random_state=42)


sc3 = StandardScaler()
X_train_smenn = sc3.fit_transform(X_train_smenn)
X_test_smenn = sc3.transform(X_test_smenn)


n_estimators = [int(x) for x in np.linspace(start=10, stop=20, num=4)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 50, num=5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# print(random_grid)

# Model training for Randomized Search CV
rand_classifier = RandomForestClassifier()
RFC_random = RandomizedSearchCV(estimator=rand_classifier, param_distributions=random_grid,
                                n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
RFC_random.fit(X_train_smenn, y_train_smenn)

joblib.dump(RFC_random, "random_forestRS.joblib")





