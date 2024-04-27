import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import streamlit as st

file_name = "Dataset_COC_Project_F.csv"
data = pd.read_csv(file_name)#loading the dataset

data.drop(['Smoke/Alcohol_Consumption', 'Past_History'], axis=1, inplace=True)#dropping the not required columns


X_text = data[['Symptom_1', 'Symptom_2', 'Symptom_3']].astype(str).apply(lambda x: ' '.join(x), axis=1)#joining all the three symptoms as strings into one list via the join fuction 
X_numeric = data[['Duration_of_Symptom', 'Age']]# storing all the numeric features in one list
y = data['Disease']#output var

le = LabelEncoder()#obj
X_numeric['Gender'] = le.fit_transform(data['Gender'])#adding a new column to the numeric list as gender by transforming it a numeric value 

severity_mapping = {'Mild': 1, 'Moderate': 2, 'Severe': 3}#preprocessing "severity" via mapping 
X_numeric['Severity'] = data['Severity'].map(severity_mapping)#adding severity to the x_numeric list 

scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)#scaling the numeric values to bring them within a range 

tfidf_vectorizer = TfidfVectorizer()
X_text_tfidf = tfidf_vectorizer.fit_transform(X_text)#applying tf-idf so that imporatnces of all the text input can be identified

X = pd.concat([pd.DataFrame(X_text_tfidf.toarray()), pd.DataFrame(X_numeric_scaled)], axis=1)#putting all the indpendent var(which have been preprocessed above) into one list

X.columns = X.columns.astype(str)#settign the column names as string