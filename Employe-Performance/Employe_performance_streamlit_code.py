#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# STREAMLIT APP (Run separately for prediction)

import streamlit as st
import pandas as pd
import joblib 

import streamlit as st
st.set_page_config(
    page_title="ML-Powered Employee Performance Predictor 📊",
    page_icon="🤖",
    layout="wide"  
)


# Sidebar with project summary
st.sidebar.title("Project Summary")

st.sidebar.markdown("""
### Objective:
The objective of this project is to analyze employee data and develop a predictive model to understand and improve employee performance. The project aims to provide actionable HR insights and assist in strategic workforce planning.

---

### Key Insights:
- **Work-Life Balance and OverTime** are strong indicators of employee performance.
- **Job Involvement** and **Years with Current Manager** have a positive correlation with performance ratings.
- Departments with lower average ratings may need process restructuring or management reviews.

---

### Machine Learning Model:
- **Model Used:** Random Forest Classifier
- **Why:** Handles both categorical and numerical data, provides feature importance, and performs well with default hyperparameters.
- **Performance:** The model achieved high accuracy and interpretability, suitable for real-world HR analytics.

---

### Techniques Used:
- Data cleaning, encoding, and preprocessing using Pandas & Scikit-learn.
- Exploratory Data Analysis (EDA) using Seaborn & Matplotlib.
- Feature engineering  correlation analysis.
- Feature importance extraction for business insight generation.

---

### Business Recommendations:
- Improve **Work-Life Balance** policies to boost productivity.
- Invest in **employee engagement programs** in departments with poor ratings.
- Offer **skill-based training programs** to low-performing employees.
- Encourage **long-term manager-employee relationships**.

---

### Deliverables:
- Cleaned and processed dataset
- Trained prediction model
- Actionable business recommendations
""")



import streamlit as st
import joblib # <--- ADD THIS IMPORT
# import pickle # <--- REMOVE OR COMMENT OUT THIS IMPORT

# Load scaler & model
@st.cache_resource
def load_model_scaler():
    # Use the joblib library to load the models
    # NOTE: The file names MUST match the new names (e.g., 'Scaler1.pkl')
    # and the file paths must be correct relative to the repository root.
    
    # Define the base directory (assuming this is correct from previous context)
    base_path = "Employe-Performance/" 

    # Load Scaler saved with Joblib
    scaler_path = base_path + "Employe_performance_Scaler1.pkl"
    scaler = joblib.load(scaler_path)

    # Load Model saved with Joblib
    model_path = base_path + "Employe_performance_trained_model1.pkl"
    model = joblib.load(model_path)
    
    return model, scaler

loaded_model, loaded_scaler = load_model_scaler()

freq_encode_maps = {
    "Gender": {'Male': 0.604, 'Female': 0.396},
    "EducationBackground": {'Life Sciences':0.41, 'Medical':0.32, 'Marketing':0.114,'Technical Degree':0.083,'Other':0.055,'Human Resources':0.018},
    "MaritalStatus": {'Married':0.457,'Single':0.32,'Divorced':0.223},
    "EmpDepartment": {'Sales':0.311,'Development':0.301,'Research & Development':0.286,'Human Resources':0.045,'Finance':0.041,'Data Science':0.017},
    "BusinessTravelFrequency": {'Travel_Rarely':0.705,'Travel_Frequently':0.185,'Non-Travel':0.11},
    "OverTime": {'No':0.706,'Yes':0.294},
    "Attrition": {'No':0.852,'Yes':0.148}
}

st.title("Employee Performance Rating Prediction 📊")

# Numeric inputs
age = st.number_input("Age", 18, 65)
distance = st.number_input("Distance From Home (km)", 1, 100)
education_level = st.number_input("Education Level (1-5): 1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor", 1, 5)
environment_sat = st.number_input("Environment Satisfaction (1-4)", 1, 4)
hourly_rate = st.number_input("Hourly Rate", 1, 500)
job_involve = st.number_input("Job Involvement (1-4)", 1, 4)
job_level = st.number_input("Job Level (1-5)", 1, 5)
job_sat = st.number_input("Job Satisfaction (1-4)", 1, 4)
num_comp_worked = st.number_input("Number Companies Worked", 1, 10)
salary_hike = st.number_input("Last Salary Hike Percent", 0, 100)
relation_sat = st.number_input("Relationship Satisfaction (1-4)", 1, 4)
work_exp = st.number_input("Total Work Experience Years", 0, 50)
training_last_year = st.number_input("Training Times Last Year", 0, 20)
work_life_bal = st.number_input("Work Life Balance (1-4)", 1, 4)
exp_current_company = st.number_input("Experience Years at Company", 0, 40)
exp_current_role = st.number_input("Experience Years in Current Role", 0, 40)
years_since_promo = st.number_input("Years Since Last Promotion", 0, 20)
years_with_manager = st.number_input("Years With Current Manager", 0, 40)
gender = st.selectbox("Gender", ['Female', 'Male'])
education_background = st.selectbox("Education Background", ['Human Resources','Life Sciences','Marketing','Medical','Other','Technical Degree'])
marital_status = st.selectbox("Marital Status", ['Divorced','Married','Single'])
emp_department = st.selectbox("Department", ['Data Science','Development','Finance','Human Resources','Research & Development','Sales'])
emp_job_role = st.number_input("EmpJobRole Score (mean rating; see docs)", 1.0, 4.0)
business_travel = st.selectbox("Business Travel Frequency", ['Non-Travel','Travel_Frequently','Travel_Rarely'])
overtime = st.selectbox("OverTime", ['No', 'Yes'])
attrition = st.selectbox("Attrition", ['No', 'Yes'])

# Prepare input dictionary
input_dict = {
    "Age": age,
    "EmpJobRole": emp_job_role,
    "DistanceFromHome": distance,
    "EmpEducationLevel": education_level,
    "EmpEnvironmentSatisfaction": environment_sat,
    "EmpHourlyRate": hourly_rate,
    "EmpJobInvolvement": job_involve,
    "EmpJobLevel": job_level,
    "EmpJobSatisfaction": job_sat,
    "NumCompaniesWorked": num_comp_worked,
    "EmpLastSalaryHikePercent": salary_hike,
    "EmpRelationshipSatisfaction": relation_sat,
    "TotalWorkExperienceInYears": work_exp,
    "TrainingTimesLastYear": training_last_year,
    "EmpWorkLifeBalance": work_life_bal,
    "ExperienceYearsAtThisCompany": exp_current_company,
    "ExperienceYearsInCurrentRole": exp_current_role,
    "YearsSinceLastPromotion": years_since_promo,
    "YearsWithCurrManager": years_with_manager,
    "Gender": freq_encode_maps['Gender'][gender],
    "EducationBackground": freq_encode_maps['EducationBackground'][education_background],
    "MaritalStatus": freq_encode_maps['MaritalStatus'][marital_status],
    "EmpDepartment": freq_encode_maps['EmpDepartment'][emp_department],
    "BusinessTravelFrequency": freq_encode_maps['BusinessTravelFrequency'][business_travel],
    "OverTime": freq_encode_maps['OverTime'][overtime],
    "Attrition": freq_encode_maps['Attrition'][attrition],
}

input_df = pd.DataFrame([input_dict])
input_df = input_df[loaded_scaler.feature_names_in_]
input_scaled = loaded_scaler.transform(input_df)
input_scaled_df = pd.DataFrame(input_scaled, columns=loaded_scaler.feature_names_in_)

if st.button("Predict Performance Rating"):
    prediction = loaded_model.predict(input_scaled_df)
    rating_map = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}
    pred_class = int(prediction[0])
    st.success(f"Predicted Employee Performance Rating: {pred_class} ({rating_map.get(pred_class, 'Unknown')})")




# Define the dictionary
rating_map = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}

# Convert dictionary to DataFrame
df = pd.DataFrame(list(rating_map.items()), columns=["Rating", "Description"])

# Display as a static table
st.table(df)
