import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Import and Explore
print("--- STEP 1: Import & Explore ---")
df = pd.read_csv('Titanic-Dataset.csv')
print("Shape:", df.shape)
print(df.info())
print("Null values:\n", df.isnull().sum())

# 2. Handle Missing Values
print("\n--- STEP 2: Handle Missing Values ---")
# Age -> median se fill
df['Age'].fillna(df['Age'].median(), inplace=True)
# Embarked -> mode se fill
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
# Cabin -> 77% null hai, drop kar dete hai
df.drop('Cabin', axis=1, inplace=True)
print("After handling nulls:", df.isnull().sum().sum(), "nulls left")

# 3. Convert Categorical to Numerical (Encoding)
print("\n--- STEP 3: Encoding ---")
# Label Encoding for Sex (male/female -> 0/1)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-Hot Encoding for Embarked (S,C,Q)
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
print("After Encoding:\n", df.head())

# 4. Normalize / Standardize
print("\n--- STEP 4: Standardization ---")
scaler = StandardScaler()
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df[num_cols].head())

# 5. Visualize Outliers & Remove
print("\n--- STEP 5: Outlier Detection ---")
plt.figure(figsize=(10,4))
plt.boxplot([df['Age'], df['Fare']], labels=['Age','Fare'])
plt.title("Outliers Before Removal")
plt.savefig('boxplot_before.png')
plt.show()

# IQR method for Fare
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
df_cleaned = df[(df['Fare'] >= lower) & (df['Fare'] <= upper)]

print(f"Before: {df.shape}, After outlier removal: {df_cleaned.shape}")

plt.figure(figsize=(10,4))
plt.boxplot([df_cleaned['Age'], df_cleaned['Fare']], labels=['Age','Fare'])
plt.title("After Outlier Removal")
plt.savefig('boxplot_after.png')
plt.show()

# Save cleaned data
df_cleaned.to_csv('cleaned_titanic.csv', index=False)
print("Cleaned dataset saved as cleaned_titanic.csv")
