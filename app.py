import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import setup as clf_setup, compare_models as compare_clf_models
from pycaret.regression import setup as reg_setup, compare_models as compare_reg_models
from pycaret.datasets import get_data

# Function to determine if the problem is classification or regression
def determine_task_type(data, target_column):
    if data[target_column].dtype in [float, int]:
        return 'regression'
    else:
        return 'classification'

st.title("PyCaret Machine Learning App")

# File uploader for data
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Columns selection
    all_columns = data.columns.tolist()
    drop_columns = st.multiselect("Select columns to drop", all_columns)
    if drop_columns:
        data.drop(columns=drop_columns, inplace=True)

    # Perform EDA
    perform_eda = st.checkbox("Perform Exploratory Data Analysis (EDA)?")
    if perform_eda:
        columns_for_eda = st.multiselect("Select columns for EDA", data.columns)
        st.write("Selected columns for EDA:")
        st.write(data[columns_for_eda].describe())

    # Handling missing values
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == 'object':
                strategy = st.radio(f"Choose strategy for handling missing values in '{col}' (categorical)", ('Mode', 'Add New Class'))
                if strategy == 'Mode':
                    data[col].fillna(data[col].mode()[0], inplace=True)
                else:
                    data[col].fillna('Unknown', inplace=True)
            else:
                strategy = st.radio(f"Choose strategy for handling missing values in '{col}' (continuous)", ('Mean', 'Median', 'Mode'))
                if strategy == 'Mean':
                    data[col].fillna(data[col].mean(), inplace=True)
                elif strategy == 'Median':
                    data[col].fillna(data[col].median(), inplace=True)
                else:
                    data[col].fillna(data[col].mode()[0], inplace=True)

    # Encoding categorical data
    encode_method = st.radio("How to encode categorical data?", ('One-Hot Encoding', 'Label Encoding'))
    if encode_method == 'Label Encoding':
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].astype('category').cat.codes
    elif encode_method == 'One-Hot Encoding':
        data = pd.get_dummies(data, drop_first=True)

    # Choose X and y
    X = st.selectbox("Select X (independent variables)", data.columns)
    y = st.selectbox("Select y (target variable)", data.columns)

    # Determine task type
    task_type = determine_task_type(data, y)

    # PyCaret setup and training
    st.write(f"Detected Task Type: {task_type}")
    if task_type == 'classification':
        clf_setup(data, target=y)
        best_model = compare_clf_models()
    else:
        reg_setup(data, target=y)
        best_model = compare_reg_models()

    # Display model report
    st.write("Best Model:")
    st.write(best_model)


