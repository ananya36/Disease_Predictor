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
