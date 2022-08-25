import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import streamlit as st
from PIL import Image


# Create title and sub
st.write("""
# ONZLA Travel Insurance Claim Site
""")

# Open and Display cover image
image = Image.open('logo2.png')
st.image(image, caption='ONZLA Logo Copyrighted', use_column_width=True)

# Get the data
df = pd.read_csv('Travel_Ins.csv')

st.subheader('Data Detailed Description:')

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

# Show statistic on the data
st.write(df.describe())

# ##Get the feature input from user

st.subheader('')
st.subheader('')
st.subheader('Check your Claim Status Right Now !!!')

Agency = st.selectbox(
    'Choose the selected Agency :',
    (df['Agency'].drop_duplicates()))
Product = st.selectbox(
    'Choose the selected Product :',
    (df['Product'].drop_duplicates()))
Destination = st.selectbox(
    'Choose the selected Destination :',
    (df['Destination'].drop_duplicates()))
Duration = st.number_input('Duration')
Commision = st.number_input('Commision')
Age = st.number_input('Age')



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

# modelling

# random forest

# SMOTE+ENN dataset
# RFC_smenn = RandomForestClassifier(random_state=42)
# RFC_smenn.fit(X_train_smenn, y_train_smenn)
# RFC_pred_smenn = RFC_smenn.predict(X_test_smenn)
# RFC_smenn_accuracy = accuracy_score(y_test_smenn, RFC_pred_smenn)
# RFC_smenn_recall = recall_score(y_test_smenn, RFC_pred_smenn)
# RFC_smenn_f1 = f1_score(y_test_smenn, RFC_pred_smenn)
# cm = confusion_matrix(y_test_smenn, RFC_pred_smenn)

#retrieve user input

result = sc3.transform([[label_encoder1.transform([Agency])[0],
                        label_encoder4.transform([Product])[0],
                        Duration,
                        label_encoder6.transform([Destination])[0],
                        Commision,
                        Age]])

# Make prediction

loaded_rf = joblib.load("./random_forestRS.joblib")
y_pred_rand = loaded_rf.predict(result)
# rand_accuracy = accuracy_score(y_test_smenn, y_pred_rand)
# rand_recall = recall_score(y_test_smenn, y_pred_rand)
# rand_f1 = f1_score(y_test_smenn, y_pred_rand)
# print(
#     f'Accuracy = {rand_accuracy:.4f}\nRecall = {rand_recall:.4f}\nF1 score = {rand_f1:.4f}\n')
# cm = confusion_matrix(y_test_smenn, y_pred_rand)
# sns.heatmap(cm, annot=True)

# le = LabelEncoder()
# le1 = LabelEncoder()
# le2 = LabelEncoder()
# features['Agency'] = le.transform(features['Agency'])
# features['Product'] = le1.transform(features['Product'])
# features['Destination'] = le2.transform(features['Destination'])
st.subheader("")
st.subheader("")
st.subheader("")
st.subheader("Claim Status : ")

if y_pred_rand == [0]:
    st.write("Not Claimable")

if y_pred_rand == [1]:
    st.write("Claimable")

# print(endresult)

# #Store the models predictions in a variables
# prediction_RF = RandomForestClassifier.predict(user_input)
# prediction_NB = GaussianNB.predict(user_input)

# st.subheader("Output")
# if prediction_RF == 1:
#     st.write("Random Forest : Positive Diabetes")
# else:
#     st.write("Random Forest : Negative Diabetes")

# if prediction_NB == 1:
#     st.write("Naive Bayes : Positive Diabetes")
# else:
#     st.write("Naive Bayes : Negative Diabetes")

# #Set a subheader and display classification
# st.subheader('Classification: ')
# st.write("Random Forest : ", prediction_RF)
# st.write("Naive Bayes: ", prediction_NB)
