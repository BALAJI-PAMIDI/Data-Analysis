#!/usr/bin/env python
# coding: utf-8

# # Data Analysis Report: Uncovering Trends in Weekly Provisional Counts of Deaths.
# 
# ## 1. Data collected from the Data Gov.
# 
# ### link : https://catalog.data.gov/dataset/weekly-counts-of-deaths-by-state-and-select-causes-2019-2020

# In[446]:


# Calling the .CSV Dataset file using the pandas Library.

import pandas as pd

# Assuming the dataset is in CSV format and located in the same directory as your script or notebook
file_path = "/Users/balajipamidi/Desktop/Final Project /Weekly_Provisional_Counts_of_Deaths_by_State_and_Select_Causes__2020-2023 (1) copy 2.csv"

# Load the dataset into a Pandas DataFrame
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame to verify that the data has been loaded correctly
df


# In[447]:


df.info()


# ## 2. Data Cleaning 
# ##  First Method  is Renaming the columns Headings

# In[448]:


#Columns Headings
df.columns


# In[449]:


df.rename(columns={"Septicemia (A40-A41)":"Septicemia",
                   'Malignant neoplasms (C00-C97)':'Malignant neoplasms',
                   'Diabetes mellitus (E10-E14)':'Diabetes mellitus',
                   'Alzheimer disease (G30)':'Alzheimer disease',
                   'Influenza and pneumonia (J09-J18)':'Influenza and pneumonia',
                   'Chronic lower respiratory diseases (J40-J47)':'Chronic lower respiratory diseases',
                   'Other diseases of respiratory system (J00-J06,J30-J39,J67,J70-J98)':'Other diseases of respiratory system',
                   'Nephritis, nephrotic syndrome and nephrosis (N00-N07,N17-N19,N25-N27)':'Nephritis',
                   'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified (R00-R99)':'not elsewhere classified',
                   'Diseases of heart (I00-I09,I11,I13,I20-I51)':'Diseases of heart',
                   'Cerebrovascular diseases (I60-I69)':'Cerebrovascular diseases',
                   'COVID-19 (U071, Multiple Cause of Death)':'COVID-19_Multiple Cause of Death',
                   'COVID-19 (U071, Underlying Cause of Death)':'COVID-19_Underlying Cause of Death'}, inplace=True)

#New columns headings
df.columns


# In[450]:


# After Renaming the columns we see the new columns names with Datatype for each column.
df.info()


# ## Second Method is Finding the missing values.

# In[451]:


# To check for Missing values
df.isnull().sum()


# ### Dropping some columns due more missing values than threshold value is 20%.

# In[452]:


# Calculate the threshold number of missing values(Here i have consider that if a column have more than 20% of missing values then the columns id dropped) )
threshold_columns = int(0.2 * df.shape[0])

# Drop columns with more than the threshold number of missing values
df.dropna(axis=1, thresh=df.shape[0] - threshold_columns, inplace=True)

# Now columns with more than 20% missing values have been dropped from the DataFrame
#df = df.drop(df.columns[df.apply(lambda col: col.isna().sum() > 2000)], axis=1)
df.shape


# In[453]:


df.info()


# ## Third Method is Imputation using to fill the missing values.

# In[454]:


#Here we have used the Mean value of each column and filled the missing values of that paticular column.
# Calculate the mean for each specified column
mean_malignant_neoplasms = df['Malignant neoplasms'].mean()
mean_alzheimer_disease = df['Alzheimer disease'].mean()
mean_chronic_respiratory_diseases = df['Chronic lower respiratory diseases'].mean()
mean_diseases_of_heart = df['Diseases of heart'].mean()
mean_cerebrovascular_diseases = df['Cerebrovascular diseases'].mean()
mean_covid_multiple_cause_of_death = df['COVID-19_Multiple Cause of Death'].mean()


# Fill missing values in each specified column with the respective mean
df['Malignant neoplasms'].fillna(mean_malignant_neoplasms, inplace=True)
df['Alzheimer disease'].fillna(mean_alzheimer_disease, inplace=True)
df['Chronic lower respiratory diseases'].fillna(mean_chronic_respiratory_diseases, inplace=True)
df['Diseases of heart'].fillna(mean_diseases_of_heart, inplace=True)
df['Cerebrovascular diseases'].fillna(mean_cerebrovascular_diseases, inplace=True)
df['COVID-19_Multiple Cause of Death'].fillna(mean_covid_multiple_cause_of_death, inplace=True)


# Now the DataFrame 'df' will have missing values in these columns filled with their respective means

df.isnull().sum()


# ## Fourth Method is Finding and deleting the Duplicate values.

# In[455]:


# Find duplicate values in both rows and columns
duplicate_values = df[df.duplicated(keep=False)]

# Display duplicate values if any are found
if not duplicate_values.empty:
    print("Duplicate Values in Both Rows and Columns:")
    print(duplicate_values)
else:
    print("No duplicate values found in both rows and columns.")
    
df


# ## Fifth Method : Finding the unique values

# In[456]:


#inding the unquie values in column the "Jurisdiction of Occurrence"

unique_values = df['Jurisdiction of Occurrence'].unique()

# Print the unique values
print(unique_values)


# ### We can see that the columns Contains all States of USA. Except ('New York' 'New York City'). We are going to replace the New York City values to New York.

# In[457]:


# Assuming df is your DataFrame with the 'Jurisdiction of Occurrence' column
# Replace df with your actual DataFrame name

# Replace 'New York City' with 'New York'
df['Jurisdiction of Occurrence'].replace('New York City', 'New York', inplace=True)
#new unique values
unique_values = df['Jurisdiction of Occurrence'].unique()
# Count the occurrences of each state
state_counts = df['Jurisdiction of Occurrence'].value_counts()

# Print the unique values and their counts
print("Counts of each state:")
# Print the unique values
print(unique_values)
print(state_counts)


# ## Sixth Method is changing the Format of the Date column
# ## Here we have changed the date format into standard format d/m/y

# In[458]:


# Assuming df is your DataFrame with columns 'Data As Of' and 'Week Ending Date'
# Replace df with your actual DataFrame name

# Convert 'Data As Of' column to desired format
df['Data As Of'] = pd.to_datetime(df['Data As Of'], errors='coerce').dt.strftime('%d/%m/%Y')

# Convert 'Week Ending Date' column to desired format
df['Week Ending Date'] = pd.to_datetime(df['Week Ending Date'], errors='coerce').dt.strftime('%d/%m/%Y')

# Now both columns are in the "day/month/year" format
df


# ## Seventh Method is changing the datatype for numeric values 
# ### from float64 to int64

# In[459]:


import pandas as pd

# Assuming df is your DataFrame containing the data
# Replace df with the name of your DataFrame

# Columns to convert from float64 to int64
columns_to_convert = [
    'Malignant neoplasms',
    'Alzheimer disease',
    'Chronic lower respiratory diseases',
    'Diseases of heart',
    'Cerebrovascular diseases',
    'COVID-19_Multiple Cause of Death'
]

# Convert each column to int64
df[columns_to_convert] = df[columns_to_convert].astype('int64')

# Verify the changes
print(df.dtypes)


# ## Eight Method is  to remove leading and trailing spaces from values in all columns 
# ### removing at column names and columns values

# In[460]:


import pandas as pd

# Example DataFrame
data = df 

df = pd.DataFrame(data)

# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove leading and trailing spaces from values in all columns
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Print the resulting DataFrame
print(df)


# In[461]:


df


# In[462]:


df.info()


# # 3. Data Manipulation
# 
# ### Performed data manipulation and feature engineering tasks on the DataFrame.
# 
# ### The code preprocesses and performs feature engineering on a DataFrame, including converting a date column to datetime format, extracting year and week information, calculating days since the last week, computing total deaths and cause-specific mortality rates, and calculating ratios, all while dropping unnecessary columns.

# In[463]:


import pandas as pd

# Assuming cleaned_df is your DataFrame with the provided columns
df = df

# Convert 'Week Ending Date' to datetime with the appropriate format
df['Week Ending Date'] = pd.to_datetime(df['Week Ending Date'], format='%d/%m/%Y')

# Feature Engineering
df['Year'] = df['Week Ending Date'].dt.year
df['Week'] = df['Week Ending Date'].dt.isocalendar().week

# Days since last week (assuming data is sorted by date)
df['Days_Since_Last_Week'] = df['Week Ending Date'].diff().dt.days.fillna(0)

# Total Deaths
# Sum up all the columns related to causes of death
df['Total_Deaths'] = df[['All Cause', 'Natural Cause', 'Malignant neoplasms', 'Alzheimer disease', 'Chronic lower respiratory diseases', 'Diseases of heart', 'Cerebrovascular diseases', 'COVID-19_Multiple Cause of Death']].sum(axis=1)

# Cause-specific Mortality Rates (examples)
df['Malignancy_Mortality_Rate'] = df['Malignant neoplasms'] / df['Total_Deaths'] * 100
df['Heart_Disease_Mortality_Rate'] = df['Diseases of heart'] / df['Total_Deaths'] * 100

# Ratio (example)
df['Natural_Cause_Prop'] = cleaned_df['Natural Cause'] / df['Total_Deaths']

df['Month'] = df['Week Ending Date'].dt.month
# Assuming 'Season' is derived from 'Month' column
df['Season'] = df['Month'].apply(lambda x: (x%12 + 3)//3)
# Identifying and encoding seasonal patterns
season_map = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Autumn'}
df['Seasonal Pattern'] = df['Season'].map(season_map)

print(df)


# In[464]:


df


# In[465]:


df.info()


# # Data Analysis 
# 
# ## Descriptive Statistics:
# 
# ### Calculate mean, median, mode, standard deviation and variance

# In[487]:


import pandas as pd

# Assuming your DataFrame is named 'df'http://localhost:8888/notebooks/Balaji_Pamidi_Final_project_002644804.ipynb#Calculate-mean,-median,-mode,-standard-deviation-and-variance

# Select numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Calculate descriptive statistics
descriptive_stats = numeric_columns.describe()

# Calculate mode
#mode = numeric_columns.mode()

# Loop through each numeric column and calculate mode separately
for column in numeric_columns:
    mode = numeric_columns[column].mode()
    print(f"Mode for {column}: {mode.tolist()}")

# Calculate variance
variance = numeric_columns.var()


# Display the results
print("Descriptive Statistics:")
print(descriptive_stats)

#print("\nMode:")
#print(mode)

print("\nVariance:")
print(variance)


# # Correlation 

# In[466]:


# Select only numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64','UInt32'])

# Calculate correlation
correlation_matrix = numeric_columns.corr()

# Display correlation matrix
print(correlation_matrix)


# Finding the strongest correlation
strongest_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()
print("\nStrongest Correlation:")
print(strongest_corr[:30])  # Displaying the top 5 strongest correlations

# Finding the moderate correlation (between 0.3 and 0.7)
moderate_corr = strongest_corr[(strongest_corr < 0.7) & (strongest_corr >= 0.3)]
print("\nModerate Correlation:")
print(moderate_corr)

# Finding the weakest correlation
weakest_corr = correlation_matrix.unstack().sort_values().drop_duplicates()
print("\nWeakest Correlation:")
print(weakest_corr[:5])  # Displaying the top 5 weakest correlations


# In[490]:


# Heatmap for correlation matrix
plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# # Multiple Linear Regression Analysis 
# ### Multiple Linear Regression using sklearn and using statsmodels

# In[467]:


import pandas as pd
import statsmodels.api as sm

# Assuming cleaned_df is your DataFrame with the provided columns
# Select predictor variables and outcome variable
predictor_cols = ['Cerebrovascular diseases','Alzheimer disease','Chronic lower respiratory diseases']

outcome_col = ['Total_Deaths']

# Create a DataFrame with predictor variables
X = df[predictor_cols]

# Add a constant term to the predictor variables (for intercept)
X = sm.add_constant(X)

# Create a Series with the outcome variable
y = df[outcome_col]

# Concatenate X and y along the columns axis
df_copy = pd.concat([X, y], axis=1)

# Print the concatenated DataFrame
print(df_copy.head(5))

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary()) 


# ## Model Building
# ### Training Models

# In[493]:


#split the dataset in training set and test set

X_train, X_test, y_train, y_test = train_test_split(df[predictor_cols], df[outcome_col], test_size=0.3, random_state=0)

# Convert y_train and y_test to DataFrame with a single column
y_train = pd.DataFrame(y_train, columns=outcome_col)
y_test = pd.DataFrame(y_test, columns=outcome_col)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape) 


# ## Using the sklearn we are going to implement the Multiple linear regression

# In[494]:


from sklearn.linear_model import LinearRegression

# Create an instance of the LinearRegression class
lin_reg = LinearRegression()

# Fit the model using the training data
lin_reg.fit(X_train, y_train)

# After fitting, you can access the coefficients and intercept of the model
coefficients = lin_reg.coef_
intercept = lin_reg.intercept_

# Print the coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)


# ### Making Predictions

# In[495]:


# Make predictions on the testing data
y_pred = lin_reg.predict(X_test)

# Print the first few predictions
print("y_pred:", y_pred[:5])


# ### Evaluating Model Predict

# In[496]:


# Use model to predict

lin_reg.predict([[3110,2537,3501]])


# ## R2_Score 

# In[497]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# ## Plot the results using matplotlib

# In[498]:


#plot the results
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.xlabel('predicted')
plt.title('Actual vs. predicted')


# ### Differnce between the test and predicted values

# In[474]:


# Convert y_test and y_pred to one-dimensional arrays
y_test_1d = y_test.values.flatten()
y_pred_1d = y_pred.flatten()

# Create DataFrame with one-dimensional arrays
pred_y_df = pd.DataFrame({'Actual Value': y_test_1d, 'Predicted Value': y_pred_1d, 'Difference': y_test_1d - y_pred_1d})
pred_y_df.head(20)


# ## Removing the Outliers

# In[475]:


import pandas as pd
import numpy as np

# Load your dataset (replace 'your_dataset.csv' with the actual path to your dataset)
df_uncleaned = df

# Make a copy of the original DataFrame to work with
df_uncleaned = df.copy()

# Specify the numerical columns in your dataset
numeric_columns = [
    
   'Cerebrovascular diseases',
    'Alzheimer disease',
    'Chronic lower respiratory diseases',
    'Total_Deaths'
]

# Function to detect outliers using IQR method
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

# Detect outliers in each numerical column
outliers = {}
for col in numeric_columns:
    outliers[col] = detect_outliers_iqr(df_uncleaned[col])

# Print count of outliers for each column
for col, is_outlier in outliers.items():
    num_outliers = sum(is_outlier)
    print(f"Outliers detected in '{col}': {num_outliers}")

# Create a copy of the DataFrame for trimming outliers
df_trimmed = df_uncleaned.copy()

# Filter out rows containing outliers
for col, is_outlier in outliers.items():
    df_trimmed = df_trimmed.loc[~is_outlier]

# Reset the index of the trimmed DataFrame
df_trimmed.reset_index(drop=True, inplace=True)

# Print count of rows after removing outliers
print(f"Count of rows after removing outliers: {len(df_trimmed)}")

# Print count of values in each column after trimming outliers
for col in df_trimmed.columns:
    print(f"Count of values in '{col}': {df_trimmed[col].count()}")


# ## Correlation after removing the outliers

# In[476]:


# Select only numeric columns
numeric_columns = df_trimmed.select_dtypes(include=['int64', 'float64','UInt32'])

# Calculate correlation
correlation_matrix = numeric_columns.corr()

# Display correlation matrix
print(correlation_matrix)


# Finding the strongest correlation
strongest_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()
print("\nStrongest Correlation:")
print(strongest_corr[:30])  # Displaying the top 5 strongest correlations

# Finding the moderate correlation (between 0.3 and 0.7)
moderate_corr = strongest_corr[(strongest_corr < 0.7) & (strongest_corr >= 0.3)]
print("\nModerate Correlation:")
print(moderate_corr)

# Finding the weakest correlation
weakest_corr = correlation_matrix.unstack().sort_values().drop_duplicates()
print("\nWeakest Correlation:")
print(weakest_corr[:5])  # Displaying the top 5 weakest correlations


# # Multiple linear regression after removing the outliers

# In[477]:


import pandas as pd
import statsmodels.api as sm

# Assuming cleaned_df is your DataFrame with the provided columns
# Select predictor variables and outcome variable
predictor_cols = ['Cerebrovascular diseases','Alzheimer disease','Chronic lower respiratory diseases']

outcome_col = ['Total_Deaths']

# Create a DataFrame with predictor variables
X = df_trimmed[predictor_cols]

# Add a constant term to the predictor variables (for intercept)
X = sm.add_constant(X)

# Create a Series with the outcome variable
y = df_trimmed[outcome_col]

# Concatenate X and y along the columns axis
df_copy = pd.concat([X, y], axis=1)

# Print the concatenated DataFrame
print(df_copy.head(5))

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())


# In[478]:


#split the dataset in training set and test set

X_train, X_test, y_train, y_test = train_test_split(df_trimmed[predictor_cols], df_trimmed[outcome_col], test_size=0.3, random_state=0)

# Convert y_train and y_test to DataFrame with a single column
y_train = pd.DataFrame(y_train, columns=outcome_col)
y_test = pd.DataFrame(y_test, columns=outcome_col)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[479]:


from sklearn.linear_model import LinearRegression

# Create an instance of the LinearRegression class
lin_reg = LinearRegression()

# Fit the model using the training data
lin_reg.fit(X_train, y_train)

# After fitting, you can access the coefficients and intercept of the model
coefficients = lin_reg.coef_
intercept = lin_reg.intercept_

# Print the coefficients and intercept
print("Coefficients:", coefficients)
print("Intercept:", intercept)


# In[480]:


# Make predictions on the testing data
y_pred = lin_reg.predict(X_test)

# Print the first few predictions
print("y_pred:", y_pred[:5])


# In[481]:


# # use model to predict
#2776
lin_reg.predict([[81,54,86]])


# In[482]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[483]:


#plot the results

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.xlabel('predicted')
plt.title('Actual vs. predicted')


# In[484]:


import seaborn as sns
import matplotlib.pyplot as plt

# Specify the column for which you want to compare outliers
column_to_check = 'All Cause'

# Detect outliers for the specified column using the IQR method
outliers_before = detect_outliers_iqr(df_uncleaned[column_to_check])

# Filter out rows containing outliers for the specified column
df_trimmed = df_trimmed.loc[~outliers_before]

# Create a box plot using Seaborn to visualize outliers before and after trimming
plt.figure(figsize=(10, 6))

# Box plot before trimming outliers
plt.subplot(1, 2, 1)
sns.boxplot(y=df_uncleaned[column_to_check], color='skyblue')
plt.title(f'Outliers Before Trimming - {column_to_check}')

# Box plot after trimming outliers
plt.subplot(1, 2, 2)
sns.boxplot(y=df_trimmed[column_to_check], color='lightgreen')
plt.title(f'Outliers After Trimming - {column_to_check}')

plt.tight_layout()
plt.show()


# In[ ]:




