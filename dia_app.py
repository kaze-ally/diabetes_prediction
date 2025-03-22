import streamlit as st
from pred_help import predict
st.title("Diabetes Predictor")

 # 1. GENDER
gender = st.selectbox("Gender", ["Male", "Female", "Others"])
# Encode gender to numeric (example encoding, adjust if needed)

# 2. AGE
age = st.number_input("Age", min_value=1, max_value=100, value=30)

# 3. HEART DISEASE
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
# Encode heart disease (Yes=1, No=0)
heart_disease_code = 1 if heart_disease == "Yes" else 0

# 4. HYPERTENSION
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
# Encode hypertension (Yes=1, No=0)
hypertension_code = 1 if hypertension == "Yes" else 0
# 5. SMOKING HISTORY
smoking_history = st.selectbox("Smoking History",["never", "ever", "former", "current", "not current"])


# 6. BMI
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=60.0, value=25.0, step=0.1)

# 7. HbA1c Level
hba1c_level = st.number_input("HbA1c Level", min_value=1.0, max_value=10.0, value=5.5, step=0.1)

# 8. Blood Glucose Level
blood_glucose = st.number_input("Blood Glucose Level", min_value=50, max_value=300, value=100, step=1)

          
input_dict = {
            "gender": gender,
            "age": age,
            "heart_disease": heart_disease_code,
            "hypertension": hypertension_code,
            "smoking_history": smoking_history,
            "bmi": bmi,
            'HbA1c_level': hba1c_level,
            "blood_glucose_level": blood_glucose
            }
            
# Button to make prediction
if st.button('Predict'):
    prediction = predict(input_dict)
    if prediction==0:
        st.success("The model predicts: **No Diabetic")
    if prediction==1:
        st.error("The model predicts: **Diabetic**")


        

        

            


