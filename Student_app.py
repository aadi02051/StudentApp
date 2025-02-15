import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open ("student_lr_final_model.pkl",'rb') as file:
        model, scaler, le = pickle.load(file)
    return model, scaler, le

def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    df = pd.DataFrame([data])
    transformed_data = scaler.fit_transform(df)
    return transformed_data

def predict_data(data):
    model, standard_scaler, LabelEncoder = load_model()
    transformed_data = preprocessing_input_data(data, standard_scaler, LabelEncoder)
    predicted_value = model.predict(transformed_data)
    return predicted_value

def main():
    st.title("Student perfomance prediction")
    st.write("enter your data to get prediction")
    
    hour_sutdied = st.number_input("Hours studied",min_value = 1, max_value = 10 , value = 5)
    prvious_score = st.number_input("previous score",min_value = 40, max_value = 100 , value = 70)
    extra = st.selectbox("extra curri activity" , ['Yes',"No"])
    sleeping_hour = st.number_input("sleeping hours",min_value = 4, max_value = 10 , value = 7)
    number_of_peper_solved = st.number_input("number of question paper solved",min_value = 0, max_value = 10 , value = 5)
    
    if st.button("predict-your_score"):
        user_data = {
            "Hours Studied":hour_sutdied,
            "Previous Scores":prvious_score,
            "Extracurricular Activities":extra,
            "Sleep Hours":sleeping_hour,
            "Sample Question Papers Practiced":number_of_peper_solved
        }
        print(user_data)
        prediction = predict_data(user_data)
        st.success(f"your prediciotn result is {prediction}")

if __name__ == "__main__":
    main()