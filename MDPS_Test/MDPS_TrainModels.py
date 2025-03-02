import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from MDPS_Preprocess import load_data, preprocess_data
from MDPS_HelperFunctions import save_model
import joblib


# Load datasets and preprocess
print("before load the dataset")
df_liver = load_data("data/indian_liver_patient.csv")
df_liver, scaler_liver = preprocess_data(df_liver, 'liver')

df_kidney = load_data("data/kidney_disease.csv")
df_kidney, scaler_kidney = preprocess_data(df_kidney, 'kidney')

df_parkinson = load_data("data/parkinsons.csv")
df_parkinson, scaler_parkinson = preprocess_data(df_parkinson, 'parkinson')

# Train liver disease model
X_liver = df_liver.drop('Dataset', axis=1)
y_liver = df_liver['Dataset']
X_train, X_test, y_train, y_test = train_test_split(X_liver, y_liver, test_size=0.3, random_state=42)

logreg_model_liver = LogisticRegression()
logreg_model_liver.fit(X_train, y_train)

# Train kidney disease model
#X_kidney = df_kidney.drop('classification', axis=1)
#y_kidney = df_kidney['classification']
X_kidney = df_kidney.drop('id', axis=1)
y_kidney = df_kidney['id']
X_train, X_test, y_train, y_test = train_test_split(X_kidney, y_kidney, test_size=0.3, random_state=42)

rf_model_kidney = RandomForestClassifier()
rf_model_kidney.fit(X_train, y_train)

# Train Parkinson's disease model
X_parkinson = df_parkinson.drop(['status', 'name'], axis=1)
y_parkinson = df_parkinson['status']
X_train, X_test, y_train, y_test = train_test_split(X_parkinson, y_parkinson, test_size=0.3, random_state=42)

svc_model_parkinson = SVC(probability=True)
svc_model_parkinson.fit(X_train, y_train)
print("before save the model")
# Save models and scalers
save_model(logreg_model_liver, 'models/liver_model.pkl')
save_model(rf_model_kidney, 'models/kidney_model.pkl')
save_model(svc_model_parkinson, 'models/parkinson_model.pkl')

joblib.dump(scaler_liver, 'models/scaler_liver.pkl')
joblib.dump(scaler_kidney, 'models/scaler_kidney.pkl')
joblib.dump(scaler_parkinson, 'models/scaler_parkinson.pkl')
