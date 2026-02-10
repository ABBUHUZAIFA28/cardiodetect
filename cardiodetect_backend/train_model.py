import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Only the 6 selected features
SELECTED_FEATURES = ['BMI', 'Age', 'GenHlth', 'PhysHlth', 'Income', 'MentHlth']
TARGET = "HeartDiseaseorAttack"

def train_model(csv_path="BRFSS.csv"):
    # Load and clean data
    df = pd.read_csv(csv_path).dropna()

    # Encode object-type columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Select only your chosen features
    X = df[SELECTED_FEATURES]
    y = df[TARGET]

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Accuracy: {acc * 100:.2f}%")

    # Save files inside model/ folder
    joblib.dump(model, "./models/final_lightgbm_model.pkl")
    joblib.dump(SELECTED_FEATURES, "./models/brfss_selected_features.pkl")
    joblib.dump(scaler, "./models/scaler.pkl")

if __name__ == "__main__":
    train_model()
