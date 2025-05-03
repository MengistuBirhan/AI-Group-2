import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import altair as alt


#  Grade Mapping Functions
 
def map_grade_to_numeric_letter(score):
    if score < 40:
        return 0.0, 'F'
    elif 40 <= score < 45:
        return 1.0, 'D'
    elif 45 <= score < 50:
        return 1.75, 'C-'
    elif 50 <= score < 60:
        return 2.0, 'C'
    elif 60 <= score < 65:
        return 2.5, 'C+'
    elif 65 <= score < 70:
        return 2.75, 'B-'
    elif 70 <= score < 75:
        return 3.0, 'B'
    elif 75 <= score < 80:
        return 3.5, 'B+'
    elif 80 <= score < 85:
        return 3.75, 'A-'
    elif 85 <= score < 90:
        return 4.0, 'A'
    else:
        return 4.1, 'A+'

def convert_numeric_to_letter(grade):
    if grade < 1.0:
        return "F"
    elif grade < 1.75:
        return "D"
    elif grade < 2.0:
        return "C-"
    elif grade < 2.5:
        return "C"
    elif grade < 2.75:
        return "C+"
    elif grade < 3.0:
        return "B-"
    elif grade < 3.5:
        return "B"
    elif grade < 3.75:
        return "B+"
    elif grade < 4.0:
        return "A-"
    elif grade <= 4.0:
        return "A"
    else:
        return "A+"

 
#  Load and Preprocess Data
 
data = pd.read_csv('Students _Performance _Prediction.csv')

categorical_columns = data.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

data[['Numeric_Grade', 'Letter_Grade']] = data['Grade'].apply(lambda x: pd.Series(map_grade_to_numeric_letter(x))
)feature_columns = ['Student_Age', 'Sex', 'High_School_Type', 'Scholarship','Additional_Work', 'Sports_activity', 'Transportation','Weekly_Study_Hours', 'Attendance', 'Reading', 'Notes', 'Listening_in_Class', 'Project_work']

X = data[feature_columns]
y = data['Numeric_Grade']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=34)

 
#  Model Selection and Loading/Saving
 
MODEL_FILE_PATH = "student_performance_model.pkl"
LABEL_ENCODER_FILE_PATH = "label_encoders.pkl"

st.sidebar.header("⚙️ Model Selection")
model_type = st.sidebar.selectbox("Select a Regression Model", ["Random Forest", "Gradient Boosting", "Linear Regression"])

# Function to load the model and label encoders

def load_model_and_encoders():
    if os.path.exists(MODEL_FILE_PATH) and os.path.exists(LABEL_ENCODER_FILE_PATH):
        with open(MODEL_FILE_PATH, 'rb') as file:
            loaded_model = pickle.load(file)
        with open(LABEL_ENCODER_FILE_PATH, 'rb') as file:
            loaded_encoders = pickle.load(file)
        return loaded_model, loaded_encoders
    return None, None

# Function to save the model and label encoders

def save_model_and_encoders(trained_model, encoders):
    with open(MODEL_FILE_PATH, 'wb') as file:
        pickle.dump(trained_model, file)
    with open(LABEL_ENCODER_FILE_PATH, 'wb') as file:
        pickle.dump(encoders, file)

loaded_model, loaded_label_encoders = load_model_and_encoders()

if loaded_model:
    model = loaded_model
    label_encoders = loaded_label_encoders
    st.sidebar.success("Loaded pre-trained model and encoders!")
else:
    if model_type == "Random Forest":
        model = RandomForestRegressor(random_state=34)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=34)
    elif model_type == "Linear Regression":
        model = LinearRegression()

    model.fit(X_train, y_train)
    save_model_and_encoders(model, label_encoders)
    st.sidebar.info("Trained and saved the model and encoders.")

 
# Feature Importance (Conditional on Tree-Based Models)
 
if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

 
#  Model Evaluation
 
y_pred_test = model.predict(X_test)
r_squared = r2_score(y_test, y_pred_test)

 
#  Advanced Data Exploration
 
st.sidebar.header("Advanced Data Exploration")

if st.sidebar.checkbox("Show Scatter Plot"):
    st.subheader("Interactive Scatter Plot")
    x_feature = st.selectbox("Select X-axis Feature", feature_columns)
    y_feature = st.selectbox("Select Y-axis Feature", feature_columns + ['Numeric_Grade'])
    if x_feature and y_feature:
        chart = alt.Chart(data).mark_circle().encode( x=x_feature,y=y_feature,tooltip=[x_feature, y_feature, 'Letter_Grade', 'Student_Age']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

 
#  Streamlit UI
 
st.title("Student Performance Predictor")
if loaded_model:
    st.markdown(f"Predicting final grade using **{type(model).__name__}** (loaded)")
else:
    st.markdown(f"Predicting final grade using **{model_type}**")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        student_age = st.number_input("Student Age", 17, 30, 20)
        sex = st.selectbox("Sex", ['Male', 'Female'])
        high_school_type = st.selectbox("High School Type", ['Public', 'Private'])
        scholarship = st.selectbox("Scholarship", [50, 75, 100])
        additional_work = st.selectbox("Additional Work", ['Yes', 'No'])
        sports_activity = st.selectbox("Sports Activity", ['Yes', 'No'])

    with col2:
        transportation = st.selectbox("Transportation", ['Private', 'Bus'])
        weekly_study_hours = st.slider("Weekly Study Hours", 0.0, 40.0, 10.0)
        attendance = st.selectbox("Attendance Score", [1.0, 2.0, 3.0])
        reading = st.selectbox("Reading Score", ['Yes', 'No'])
        notes = st.selectbox("Notes Score", [1.0, 0.0])
        listening_in_class = st.selectbox("Listening in Class", [1.0, 0.0])
        project_work = st.selectbox("Project Work", [1.0, 0.0])

    submit = st.form_submit_button("Predict Grade")

if submit:
    input_data = {
        'Student_Age': student_age,
        'Sex': sex,
        'High_School_Type': high_school_type,
        'Scholarship': scholarship,
        'Additional_Work': additional_work,
        'Sports_activity': sports_activity,
        'Transportation': transportation,
        'Weekly_Study_Hours': weekly_study_hours,
        'Attendance': attendance,
        'Reading': reading,
        'Notes': notes,
        'Listening_in_class': listening_in_class,
        'Project_work': project_work
    }

    input_df = pd.DataFrame([input_data])

    # Encode input using loaded or newly fitted encoders
    encoders_to_use = loaded_label_encoders if loaded_label_encoders else label_encoders
    for col in input_df.columns:
        if col in encoders_to_use:
            encoder = encoders_to_use[col]
            try:
                input_df[col] = encoder.transform(input_df[col])
            except ValueError:
                input_df[col] = input_df[col].apply(
                    lambda x: encoder.transform([encoder.classes_[0]])[0]
                    if x not in encoder.classes_ else encoder.transform([x])[0]
                )

    input_df = input_df[X.columns]

# Predict

    pred_numeric = model.predict(input_df)[0]
    pred_letter = convert_numeric_to_letter(pred_numeric)

    st.subheader("Prediction Result")
    st.success(f" Predicted Grade: **{pred_numeric:.2f} ➝ {pred_letter}**")

 # Display Feature Importance (Conditional)
 
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        st.subheader("Feature Importance")
        st.markdown("Relative importance of each factor in the prediction.")
        fig_importance, ax_importance = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax_importance)
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        st.pyplot(fig_importance)

#  Display Model Evaluation

st.subheader("Model Evaluation")
st.markdown("Performance of the model on unseen data.")
st.metric("R-squared Score", f"{r_squared:.3f}")

# Data Exploration

st.sidebar.header("Data Exploration")
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(data)

if st.sidebar.checkbox("Show Data Statistics"):
    st.subheader("Data Statistics")
    st.dataframe(data.describe())

st.sidebar.subheader("Visualizations")
feature_to_plot = st.sidebar.selectbox("Select a Feature for Histogram", feature_columns + ['Numeric_Grade'])
  if st.sidebar.checkbox(f"Show Histogram of {feature_to_plot}"):
      st.subheader(f"Histogram of {feature_to_plot}")
      fig_hist, ax_hist = plt.subplots()
      sns.histplot(data[feature_to_plot], bins=20, kde=True, ax=ax_hist)
       st.pyplot(fig_hist)

categorical_feature_to_plot = st.sidebar.selectbox("Select a Categorical Feature for Count Plot", categorical_columns)
      if st.sidebar.checkbox(f"Show Count Plot of {categorical_feature_to_plot}"):
          st.subheader(f"Count Plot of {categorical_feature_to_plot}")
          fig_count, ax_count = plt.subplots()
          sns.countplot(data=data, y=categorical_feature_to_plot, ax=ax_count)
          st.pyplot(fig_count)
