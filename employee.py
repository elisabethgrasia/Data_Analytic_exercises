import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


import kagglehub

# Download latest version
path = kagglehub.dataset_download("tawfikelmetwally/employee-dataset")

print("Path to dataset files:", path)

# Load the dataset
df = pd.read_csv(os.path.join(path, "Employee.csv"))
print("Dataset loaded successfully.")

# 1. Data Preparation
df.head()
df.info()

df.isnull().sum()  # no missing values
print(df.dtypes)
print(df['Education'].unique())
print(df.duplicated().sum())   

# 2. Exploratory Data Analysis (EDA)
# 2.1 Summary Statistics
summary_stats = df.describe()
print("Summary Statistics:", summary_stats)

# Age data distribution
df['Age'].hist(bins=20, edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# 2.2 Correlation Analysis
correlation_matrix = df.corr()
print("Correlation Matrix:", correlation_matrix)


## Compare te salary averages across categories
sns.boxplot(x=df['Gender'], y=df['PaymentTier'])
plt.title("Salary Distribution by Gender")
plt.show()

df.groupby('Gender')['PaymentTier'].mean()

# 3. Visualization
# 3.1 Salary Distribution
## Compare te salary averages across Education levels
sns.boxplot(x=df['Education'], y=df['PaymentTier'])
plt.title("Salary Distribution by Education")
plt.show()

df.groupby('Education')['PaymentTier'].mean()


# 3.1.1 Heatmap correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# 3.2 Check the attrition trends
print(df["LeaveOrNot"].value_counts())

# 4. Key Questions
# 4.1 Does the higher education correlate with salary or job retention? Not related to degree. Bachelor and Master degree holders more likely to stay in the company. 
sns.boxplot(x=df['Education'], y=df['LeaveOrNot'])
plt.title("Job Retention by Education Level")
plt.show()

# 4.2 Are there gender disparities in salary or promotion? Female has more variability in salary.
# 4.3 What workplace factors (e.g., environment, colleagues) most impact employee satisfaction? Does not exist in the dataset.