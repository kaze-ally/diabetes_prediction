import pandas as pd
import joblib
model_xgb=joblib.load("artifacts/model_xgb.joblib")
scaler_col=joblib.load("artifacts/scaler_col.joblib")
def calc_smoking(smoking_hist):
        smoking_mapping = {
            'never': 0,
            'current': 2,
            'former': 1,
            'ever': 1,
            'not current': 1
        }
        smoking_history = smoking_mapping.get('smoking_history', 'never')
        if smoking_history in smoking_mapping:
            return smoking_mapping[smoking_history]
        else:
            return 0  # default value if smoking_history is not in the mapping

def handle_scaling(df):
    scaler_object = scaler_col
    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    return df



def preprocess_input(input_dict):
    expected_columns=['age', 'hypertension','heart_disease','smoking_history','bmi', 'HbA1c_level','blood_glucose_level','gender_Male','gender_Other']
    df = pd.DataFrame(0, columns=expected_columns, index=[0])
    for key, value in input_dict.items():
        if key == 'gender' and value == 'Male':
            df['gender_Male'] = 1
        elif key=="Gender" and value=="Other":
            df["gender_Other"] =1
        elif key in expected_columns:
            df[key]=value
    df["smoking_history"]=calc_smoking(input_dict["smoking_history"])
    df=handle_scaling(df)
    return df

def predict(input_dict):
    input_df=preprocess_input(input_dict)
    prediction=model_xgb.predict(input_df)
    threshold = 0.21364881 # Adjust this threshold to achieve desired recall
    prediction_proba = model_xgb.predict_proba(input_df)[:, 1]
    prediction = (prediction_proba >= threshold).astype(int)
    return prediction[0]

    



