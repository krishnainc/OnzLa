import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

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
st.image(image, caption='ML', use_column_width=True)

# Get the data
df = pd.read_csv('Travel_Ins.csv')

# Set subheader
st.subheader('Data Information:')

# Show the data as a table
st.dataframe(df)

st.subheader('Data Detailed Description:')

# Show statistic on the data
st.write(df.describe())

# Rename the column to remove spaces
df.rename(columns={'Agency Type': 'Type', 'Distribution Channel': 'Channel', 'Product Name': 'Product',
          'Net Sales': 'Net_Sales', 'Commision (in value)': 'Commision'}, inplace=True)

# separate the column
# Separate the column into numerical and categorical
numerical = ['Duration', 'Net_Sales', 'Commision', 'Age']
categorical = ['Agency', 'Type', 'Channel',
               'Product', 'Claim', 'Destination', 'Gender']
df = df[numerical+categorical]

st.subheader('Density and distribution plot for Numerical column:')

# density and Distribution plot for Numerical column
sns.set(rc={'figure.figsize': (25, 12)})
for i in numerical:
    fig = plt.figure()
    sns.distplot(df[i], hist=True, kde=True,
                 bins=30, color='darkblue',
                 hist_kws={'edgecolor': 'black'},
                 kde_kws={'linewidth': 3})
    st.pyplot(fig)

df = df.drop(['Gender'], axis=1)

# boxplot to further detect the outliers/abnormal data.
st.subheader('Outliers/Abnormal Data Detection')
sns.set(rc={'figure.figsize': (15, 6)})
for column in numerical:
    fig1 = plt.figure()
    sns.boxplot(x=df[column])
    st.pyplot(fig1)


list_1 = df[(df['Age'] == 118)].index
df.drop(list_1, inplace=True)
list_2 = df[(df['Duration'] > 4000)].index
df.drop(list_2, inplace=True)
list_3 = df[(df['Duration'] == 0)].index
df.drop(list_3, inplace=True)

# ##Get the feature input from user

st.subheader('')
st.subheader('')
st.subheader('Check your Claim Status Right Now !!!')

def get_user_input():
    Agency = st.selectbox(
        'Select your prefereed Agency :',
        (df['Agency'].drop_duplicates()))
    Product = st.selectbox(
        'Select your prefereed Product :',
        (df['Product'].drop_duplicates()))

    Destination = st.selectbox(
        'Select your prefereed Destination :',
        ('Singapore','Malaysia','Thailand'))

    Duration = st.slider('Duration', 0, 5000, 0)

    Commision = st.slider('Commision', 0, 300, 0)
    Age = st.slider('Age', 0, 120, 0)

    #AGENCY
    if Agency == 'ADM':
        Agency = 0
    elif Agency == 'ART':
        Agency = 1
    elif Agency == 'C2B':
        Agency = 2
    elif Agency == 'CBH':
        Agency = 3
    elif Agency == 'CCR':
        Agency = 4
    elif Agency == 'CSR':
        Agency = 5
    elif Agency == 'CWT':
        Agency = 6
    elif Agency == 'EPX':
        Agency = 7
    elif Agency == 'JWT':
        Agency = 8
    elif Agency == 'JZI':
        Agency = 9
    elif Agency == 'KML':
        Agency = 10
    elif Agency == 'LWC':
        Agency = 11
    elif Agency == 'RAB':
        Agency = 12
    elif Agency == 'SSI':
        Agency = 13
    elif Agency == 'TST':
        Agency = 14
    elif Agency == 'TTW':
        Agency = 15

    #PRODUCT
    if Product == '1 way Comprehensive Plan':
        Product = 0
    if Product == '2 way Comprehensive Plan':
        Product = 1
    if Product == '24 Protect':
        Product = 2
    if Product == 'Annual Gold Plan':
        Product = 3
    if Product == 'Annual Silver Plan':
        Product = 4
    if Product == 'Annual Travel Protect Gold':
        Product = 5
    if Product == 'Annual Travel Protect Platinum':
        Product = 6
    if Product == 'Annual Travel Protect Silver':
        Product = 7
    if Product == 'Basic Plan':
        Product = 8
    if Product == 'Bronze Plan':
        Product = 9
    if Product == 'Cancellation Plan':
        Product = 10
    if Product == 'Child Comprehensive Plan':
        Product = 11
    if Product == 'Comprehensive Plan':
        Product = 12
    if Product == 'Gold Plan':
        Product = 13
    if Product == 'Individual Comprehensive Plan':
        Product = 14
    if Product == 'Premier Plan':
        Product = 15
    if Product == 'Rental Vehicle Excess Insurance':
        Product = 16
    if Product == 'Silver Plan':
        Product = 17
    if Product == 'Single Trip Travel Protect Gold':
        Product = 18
    if Product == 'Single Trip Travel Protect Platinum':
        Product = 19
    if Product == 'Single Trip Travel Protect Silver':
        Product = 20
    if Product == 'Spouse or Parents Comprehensive Plan':
        Product = 21
    if Product == 'Ticket Protector':
        Product = 22
    if Product == 'Travel Cruise Protect':
        Product = 23
    if Product == 'Travel Cruise Protect Family':
        Product = 24
    if Product == 'Value Plan':
        Product = 25
    
    
    if Destination == 'Malaysia':
        Destination = 78
    elif Destination == 'Thailand':
        Destination = 128
    elif Destination == 'Singapore':
        Destination = 117

    
    


    # Store dictionary into a variable
    user_data = {'Agency': Agency,
                'Product': Product,
                'Duration': Duration,
                'Destination': Destination,
                'Commision': Commision,
                'Age': Age,
                }

    encodeddf = pd.DataFrame(data = user_data, index=[0])

    #print(encodeddf)

    return encodeddf

#Store the user input into variable
user_input = get_user_input()


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

# Correlation table
st.subheader('Correlation Table')
fig2 = plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), square=True, annot=True, cmap='coolwarm')
st.pyplot(fig2)

# feature selection
df = df.drop(['Channel'], axis=1)
df = df.drop(['Type'], axis=1)
df = df.drop(['Net_Sales'], axis=1)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# oversampling
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X, y)

#oversampling + undersampling
smenn = SMOTEENN(random_state=42)
X_smenn, y_smenn = smenn.fit_resample(X, y)

# spliting
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(
    X_sm, y_sm, test_size=0.2, random_state=42)
X_train_smenn, X_test_smenn, y_train_smenn, y_test_smenn = train_test_split(
    X_smenn, y_smenn, test_size=0.2, random_state=42)

# scaling
sc1 = StandardScaler()
X_train_ori = sc1.fit_transform(X_train)
X_test_ori = sc1.transform(X_test)

sc2 = StandardScaler()
X_train_sm = sc2.fit_transform(X_train_sm)
X_test_sm = sc2.transform(X_test_sm)

sc3 = StandardScaler()
X_train_smenn = sc3.fit_transform(X_train_smenn)
X_test_smenn = sc3.transform(X_test_smenn)

# modelling

# random forest
# Original dataset
RFC = RandomForestClassifier(random_state=42)
RFC.fit(X_train_ori, y_train)
RFC_pred_ori = RFC.predict(X_test_ori)
RFC_ori_accuracy = accuracy_score(y_test, RFC_pred_ori)
RFC_ori_recall = recall_score(y_test, RFC_pred_ori)
RFC_ori_f1 = f1_score(y_test, RFC_pred_ori)
st.subheader('Modelling (Best Fit Model)')
st.subheader('')
st.text('Random Forest Classifier (Without SMOTE)')
st.caption(f'Accuracy = {RFC_ori_accuracy:.4f}')
st.caption(f'Recall = {RFC_ori_recall:.4f}')
st.caption(f'F1 score = {RFC_ori_f1:.4f}')
st.text('Confusion Matrix')
cm = confusion_matrix(y_test, RFC_pred_ori)
fig3 = plt.figure(figsize=(20, 12))
sns.heatmap(cm, annot=True)
st.pyplot(fig3)

# SMOTE dataset
RFC_sm = RandomForestClassifier(random_state=42)
RFC_sm.fit(X_train_sm, y_train_sm)
RFC_pred_sm = RFC_sm.predict(X_test_sm)
RFC_sm_accuracy = accuracy_score(y_test_sm, RFC_pred_sm)
RFC_sm_recall = recall_score(y_test_sm, RFC_pred_sm)
RFC_sm_f1 = f1_score(y_test_sm, RFC_pred_sm)
st.subheader('')
st.text('Random Forest Classifier (SMOTE)')
st.caption(f'Accuracy = {RFC_sm_accuracy:.4f}')
st.caption(f'Recall = {RFC_sm_recall:.4f}')
st.caption(f'F1 score = {RFC_sm_f1:.4f}')
st.text('Confusion Matrix')
cm = confusion_matrix(y_test_sm, RFC_pred_sm)
fig4 = plt.figure(figsize=(20, 12))
sns.heatmap(cm, annot=True)
st.pyplot(fig4)

# SMOTE+ENN dataset
RFC_smenn = RandomForestClassifier(random_state=42)
RFC_smenn.fit(X_train_smenn, y_train_smenn)
RFC_pred_smenn = RFC_smenn.predict(X_test_smenn)
RFC_smenn_accuracy = accuracy_score(y_test_smenn, RFC_pred_smenn)
RFC_smenn_recall = recall_score(y_test_smenn, RFC_pred_smenn)
RFC_smenn_f1 = f1_score(y_test_smenn, RFC_pred_smenn)
st.subheader('')
st.text('Random Forest Classifier (SMOTE + ENN)')
st.caption(f'Accuracy = {RFC_smenn_accuracy:.4f}')
st.caption(f'Recall = {RFC_smenn_recall:.4f}')
st.caption(f'F1 score = {RFC_smenn_f1:.4f}')
st.text('Confusion Matrix')
cm = confusion_matrix(y_test_smenn, RFC_pred_smenn)
fig5 = plt.figure(figsize=(20, 12))
sns.heatmap(cm, annot=True)
st.pyplot(fig5)

# le = LabelEncoder()
# le1 = LabelEncoder()
# le2 = LabelEncoder()
features = user_input
# features['Agency'] = le.transform(features['Agency'])
# features['Product'] = le1.transform(features['Product'])
# features['Destination'] = le2.transform(features['Destination'])
target = df['Claim']
print(features)
endresult = RFC_smenn.predict(features)

st.subheader("Claim Status : ")
st.subheader("")

if endresult == [0]:
    st.write("Not Claimable")

if endresult == [1]:
    st.write("Claimable")

print(endresult)

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
