import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv("diabetes.csv")

# ==========================
# Exploratory Data Analysis
# ==========================

# print(df.shape)
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())
# print(df.duplicated().sum())
# print(df["diabetes"].value_counts())

# Remove duplicate rows
df = df.drop_duplicates()

# Check minimum values
# print(df["age"].min())
# print(df["bmi"].min())
# print(df["blood_glucose_level"].min())
# print(df["HbA1c_level"].min())

# Check dataset after cleaning
# print("Shape after removing duplicates:", df.shape)
# print(df["diabetes"].value_counts())
# print(df["diabetes"].value_counts(normalize=True))
# print(df["gender"].value_counts())
# print(df["smoking_history"].value_counts())

# ==========================
# Feature Engineering
# ==========================

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X = pd.get_dummies(
    X,
    columns=["gender", "smoking_history"],
    drop_first=True,
    dtype=int
)

# print(X.head())
# print(X.shape)

# ==========================
# Train-Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

numeric_columns = [
    "age",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_columns] = scaler.fit_transform(
    X_train[numeric_columns]
)

X_test_scaled[numeric_columns] = scaler.transform(
    X_test[numeric_columns]
)

# ==========================
# Verification (Optional)
# ==========================

# print("Training:", X_train.shape)
# print("Testing :", X_test.shape)
# print(X_train_scaled[numeric_columns].describe())
# print(y_train.value_counts(normalize=True))
# print(y_test.value_counts(normalize=True))

# Correlation Analysis
# corr = df.corr(numeric_only=True)
# print(corr["diabetes"].sort_values(ascending=False))

print("Preprocessing completed successfully.")