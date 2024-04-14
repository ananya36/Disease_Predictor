import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import streamlit as st

file_name = "Dataset_COC_Project_F.csv"
data = pd.read_csv(file_name)#loading the dataset

