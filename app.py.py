import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# Title and description
st.title("❤️ Heart Disease Prediction App")
st.write("""
This app predicts whether you might have heart disease based on health parameters.

""")

# Load dataset with target column
heart_df_full = pd.read_csv('heart.csv')  # This includes the 'target' column

# Pie chart of heart disease distribution
st.subheader("💡 Heart Disease Dataset Distribution")
counts = heart_df_full['target'].value_counts()
labels = ['No Heart Disease', 'Heart Disease']

fig, ax = plt.subplots()
ax.pie(counts, labels=labels, autopct='%0.2f%%', startangle=90, colors=['#66b3ff','#ff6666'])
ax.set_title("Heart Disease Distribution in Dataset")
ax.axis('equal')

st.pyplot(fig)


# Sidebar for user inputs
st.sidebar.header("Please enter your health details:")


def user_input_features():
    age = st.sidebar.number_input('Age', min_value=1, max_value=120, value=30)
    sex = st.sidebar.selectbox('Sex', options={0: 'Female', 1: 'Male'})
    cp = st.sidebar.selectbox('Chest Pain Type (0-3)', options=[0, 1, 2, 3])
    trestbps = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', min_value=50, max_value=250, value=120)
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', options={0: 'No', 1: 'Yes'})
    restecg = st.sidebar.selectbox('Resting Electrocardiographic Results (0-2)', options=[0, 1, 2])
    thalach = st.sidebar.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', options={0: 'No', 1: 'Yes'})
    oldpeak = st.sidebar.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0,
                                      format="%.1f")
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment (0-2)', options=[0, 1, 2])
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-3)', options=[0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect)', options=[0, 1, 2])

    data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Load the original dataset (for encoding consistency)
heart_data = pd.read_csv('heart.csv').drop(columns=['target'])

# Combine input data with original data to apply one-hot encoding
combined_df = pd.concat([input_df, heart_data], axis=0)

# One-hot encode categorical columns as in training
combined_df = pd.get_dummies(combined_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

# Keep only the first row (the user's input)
model_input = combined_df.iloc[:1, :]

# Load the model and columns used during training
model, model_columns = pickle.load(open('Random_forest_model.pkl', 'rb'))

# Reindex to match model's training columns, fill missing with 0
model_input = model_input.reindex(columns=model_columns, fill_value=0)

# Make prediction and prediction probability
prediction = model.predict(model_input)[0]
prediction_proba = model.predict_proba(model_input)[0][prediction]

# Show user input summary
st.subheader("Your Input:")
st.write(input_df)

# Show prediction result in friendly way
st.subheader("Prediction Result:")

if prediction == 1:
    st.error(
        f"Based on the information provided, there **is a risk** of heart disease with a confidence of {prediction_proba:.2%}. Please consult a healthcare professional for further evaluation.")
else:
    st.success(
        f"Based on the information provided, there **is no significant risk** of heart disease with a confidence of {prediction_proba:.2%}. However, maintain a healthy lifestyle and regular checkups.")

# Optional: Show prediction probabilities breakdown
st.subheader("Prediction Probability Details:")
st.write(f"Probability of No Heart Disease: {model.predict_proba(model_input)[0][0]:.2%}")
st.write(f"Probability of Heart Disease: {model.predict_proba(model_input)[0][1]:.2%}")

