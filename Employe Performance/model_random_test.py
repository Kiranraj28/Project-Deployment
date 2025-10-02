import pickle
import numpy as np
import pandas as pd

# Load scaler and model
with open("Employe_performance_Scaler.pkl", "rb") as f_scaler:
    scaler = pickle.load(f_scaler)
with open("Employe_performance_trained_model.pkl", "rb") as f_model:
    model = pickle.load(f_model)

# Frequency encoding maps from the original code
freq_encode_maps = {
    "Gender": {'Male': 0.604, 'Female': 0.396},
    "EducationBackground": {'Life Sciences':0.41, 'Medical':0.32, 'Marketing':0.114,'Technical Degree':0.083,'Other':0.055,'Human Resources':0.018},
    "MaritalStatus": {'Married':0.457,'Single':0.32,'Divorced':0.223},
    "EmpDepartment": {'Sales':0.311,'Development':0.301,'Research & Development':0.286,'Human Resources':0.045,'Finance':0.041,'Data Science':0.017},
    "BusinessTravelFrequency": {'Travel_Rarely':0.705,'Travel_Frequently':0.185,'Non-Travel':0.11},
    "OverTime": {'No':0.706,'Yes':0.294},
    "Attrition": {'No':0.852,'Yes':0.148}
}

# Define ranges for numeric inputs based on original code
def random_int(low, high, size=1):
    return np.random.randint(low, high+1, size)

def random_float(low, high, size=1):
    return np.random.uniform(low, high, size)

# Generate 100 random samples
num_samples = 100
data = []

for _ in range(num_samples):
    sample = {
        "Age": random_int(18, 65)[0],
        "EmpJobRole": np.round(np.random.uniform(1.0, 4.0), 2),
        "DistanceFromHome": random_int(1, 100)[0],
        "EmpEducationLevel": random_int(1, 5)[0],
        "EmpEnvironmentSatisfaction": random_int(1, 4)[0],
        "EmpHourlyRate": random_int(1, 500)[0],
        "EmpJobInvolvement": random_int(1, 4)[0],
        "EmpJobLevel": random_int(1, 5)[0],
        "EmpJobSatisfaction": random_int(1, 4)[0],
        "NumCompaniesWorked": random_int(1, 10)[0],
        "EmpLastSalaryHikePercent": random_int(0, 100)[0],
        "EmpRelationshipSatisfaction": random_int(1, 4)[0],
        "TotalWorkExperienceInYears": random_int(0, 50)[0],
        "TrainingTimesLastYear": random_int(0, 20)[0],
        "EmpWorkLifeBalance": random_int(1, 4)[0],
        "ExperienceYearsAtThisCompany": random_int(0, 40)[0],
        "ExperienceYearsInCurrentRole": random_int(0, 40)[0],
        "YearsSinceLastPromotion": random_int(0, 20)[0],
        "YearsWithCurrManager": random_int(0, 40)[0],
        "Gender": np.random.choice(list(freq_encode_maps['Gender'].values())),
        "EducationBackground": np.random.choice(list(freq_encode_maps['EducationBackground'].values())),
        "MaritalStatus": np.random.choice(list(freq_encode_maps['MaritalStatus'].values())),
        "EmpDepartment": np.random.choice(list(freq_encode_maps['EmpDepartment'].values())),
        "BusinessTravelFrequency": np.random.choice(list(freq_encode_maps['BusinessTravelFrequency'].values())),
        "OverTime": np.random.choice(list(freq_encode_maps['OverTime'].values())),
        "Attrition": np.random.choice(list(freq_encode_maps['Attrition'].values())),
    }
    data.append(sample)

df = pd.DataFrame(data)

# Reorder columns to match scaler's expected input
df = df[scaler.feature_names_in_]

# Scale inputs
X_scaled = scaler.transform(df)

# Predict
predictions = model.predict(X_scaled)

# Print distribution of predicted classes
unique, counts = np.unique(predictions, return_counts=True)
print("Prediction class distribution for 100 random samples:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples")
