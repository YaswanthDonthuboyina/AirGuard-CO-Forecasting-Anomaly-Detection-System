#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_rel



# ### Step 1: Data Acquisition and Preprocessing

# ### 1.1 Load and Inspect Dataset

# In[75]:


# data = pd.read_excel("D:/Fall2024/SaiRam/FDS/air+quality/AirQualityUCI.xlsx")


# In[ ]:


data = pd.read_excel("C:/Users/yaswa/OneDrive/Desktop/FDS/project/AirQualityUCI.xlsx")


# In[76]:


data.head(10)


# In[77]:


data.shape


# In[78]:


data.describe(include = "all")


# In[79]:


data.info()


# ### 1.2 Handle Missing Values
# In[82]:


data.replace(-200, np.nan, inplace=True)
data.interpolate(method="linear", inplace=True)


# ### Convert Date and Time columns to a datetime format for seasonal and temporal analysis.

# In[83]:


data['Date'] = data['Date'].astype(str)
data['Time'] = data['Time'].astype(str)
# Combine Date and Time columns and convert to datetime
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')


# In[84]:


data = data.drop(['Date', 'Time'], axis=1)


# ## 2. Exploratory Data Analysis (EDA)

# In[ ]:





# ### Trend Analysis Over Time

# Objective: 
# To identify whether CO and other pollutants exhibit temporal trends, such as daily peaks (due to traffic) or seasonal variations (like winter-related pollution due to heating).

# In[71]:


data.info()


# In[85]:


# Plot CO levels over time
#data_graph = data.set_index('DateTime', inplace=True)
data.set_index('DateTime', inplace=True)
#---
# This section seems to have an issue with setting the index. 
# The index is already set in the previous cell. Let's remove this redundant line.
#---
# data.set_index('DateTime', inplace=True)
plt.figure(figsize=(12, 6))
plt.plot(data['CO(GT)'], label='CO(GT)')
plt.title("CO Levels Over Time")
plt.xlabel("Date")
plt.ylabel("CO Levels (GT)")
plt.legend()
plt.show()


# In[86]:


# Rolling average (24-hour) for trend smoothing
data['CO_Rolling'] = data['CO(GT)'].rolling(window=24).mean()  # 24 hours for weekly smoothing
plt.plot(data['CO_Rolling'], label='24-hours Rolling Average', color='orange')
plt.legend()
plt.show()


#  To capture how pollutant levels fluctuate by the hour or day of the week, potentially correlating with human activities such as rush hours or industrial operation cycles.

# In[87]:


import seaborn as sns

# Hourly box plot
data['Hour'] = data.index.hour
plt.figure(figsize=(12, 6))
sns.boxplot(x='Hour', y='CO(GT)', data=data)
plt.title("CO Levels by Hour of the Day")
plt.xlabel("Hour")
plt.ylabel("CO Levels (GT)")
plt.show()


# In[88]:


# Daily averages
daily = data.groupby(data.index.date)['CO(GT)'].mean()
plt.plot(daily.index, daily.values)
plt.title("Average Daily CO Levels")
plt.xlabel("Day")
plt.ylabel("Average CO Levels")
plt.show()


# ### Seasonal Patterns

# To determine if specific seasons or months are associated with higher pollution levels, which could be due to weather conditions or seasonal human activity.

# In[89]:


# Monthly grouping and averaging
data['Month'] = data.index.month
monthly_avg = data.groupby('Month')['CO(GT)'].mean()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg.index, monthly_avg.values)
plt.title("Average Monthly CO Levels")
plt.xlabel("Month")
plt.ylabel("Average CO Levels")
plt.show()


# In[90]:


data['hour'] = data.index.hour
data.groupby('hour')['CO(GT)'].mean().plot(title="Average CO Levels by Hour of Day")
plt.show()


# ### Correlations

# In[91]:


numeric_df = data.select_dtypes(include=['float64', 'int64'])
# Calculate the correlation matrix
correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, 
            annot=True,
            fmt='.2f',
            annot_kws ={"size":9, "color": "black", "weight": "bold"},
            cmap="coolwarm", cbar_kws={"shrink": 0.8})
plt.title("Correlation Between CO(GT) and Other Features")
plt.show()


# In[220]:


correlation_matrix


# **From the above correlation matrix, we can see that **PT08.S1(CO), C6H6(GT), PT08.S2(NMHC), NOx(GT) ,PT08.S5(O3)** are highly correlated with CO(GT)**

# **The Average CO levels increase during July to October and then decrease during rest of the months.**
# **The Average CO levels are high during the morning time in between 9 and 10. and also there is an increasing pattern in between evening 4 pm to 8pm**

# ### Missing values Inputation

# In[92]:


# Impute remaining missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
data.iloc[:, :] = imputer.fit_transform(data)


# Outlier Detection and Removal

# In[93]:


from scipy.stats import zscore

# Remove outliers based on z-score for key pollutants
z_scores = np.abs(zscore(data[['CO(GT)', 'NOx(GT)', 'NO2(GT)']]))
data = data[(z_scores < 3).all(axis=1)]

# IQR-based outlier removal for all numeric columns
numeric_cols = data.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]


# In[98]:


data.shape


# ### 3. Feature Engineering

# In[95]:


data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
# Note: 'Hour' and 'Month' were already created during EDA, re-creating them here is redundant but harmless.
data['Hour'] = data.index.hour
data['Month'] = data.index.month


# #### Interaction Terms for Pollutants
# In[99]:


# Interaction terms
data['NOx_NO2'] = data['NOx(GT)'] * data['NO2(GT)']
data['NMHC_Benzene'] = data['NMHC(GT)'] * data['C6H6(GT)']

# Drop rows with NaN values that might have been created by interpolation at the edges
data.dropna(inplace=True)

# ### 4. Model Development and Evaluation
# Define features (X) and target (y)
X = data.drop(columns=['CO(GT)'])
y = data['CO(GT)']

# Split data into training and test sets BEFORE cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features as proposed
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to dataframes for clarity
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# #### 4.1 K-Fold Cross-Validation and Hyperparameter Tuning

# In[100]:

print("--- Starting Model Training and Cross-Validation ---")
# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Perform 5-fold cross-validation for baseline models
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    cv_scores[name] = scores
    print(f"{name} R^2 (mean): {np.mean(scores):.4f} (std: {np.std(scores):.4f})")

# Hyperparameter tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

print(f"Best parameters for Random Forest: {grid_search.best_params_}")
best_rf_model = grid_search.best_estimator_
cv_scores['Tuned Random Forest'] = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5, scoring='r2')
print(f"Tuned Random Forest R^2 (mean): {np.mean(cv_scores['Tuned Random Forest']):.4f} (std: {np.std(cv_scores['Tuned Random Forest']):.4f})")


# #### 4.2 Statistical Validation
# Perform a paired t-test between Linear Regression and Tuned Random Forest scores

print("\n--- Performing Statistical Validation ---")
lr_scores = cv_scores['Linear Regression']
rf_scores = cv_scores['Tuned Random Forest']
t_stat, p_value = ttest_rel(lr_scores, rf_scores)

print(f"Paired t-test between Linear Regression and Tuned Random Forest:")
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
if p_value < 0.05:
    print("The difference in performance is statistically significant.")
else:
    print("The difference in performance is not statistically significant.")


# #### 4.3 Out-of-Distribution (OOD) Detection and Testing

print("\n--- Out-of-Distribution (OOD) Detection and Testing ---")
# 1. Create OOD test set by adding Gaussian noise
noise_factor = 0.5 
X_test_ood = X_test_scaled.copy()
for col in X_test_ood.columns:
    if X_test_ood[col].std() > 0:
        noise = np.random.normal(0, noise_factor * X_test_ood[col].std(), X_test_ood.shape[0])
        X_test_ood[col] += noise

# 2. Train an OOD Detector (Isolation Forest)
ood_detector = IsolationForest(contamination=0.1, random_state=42)
ood_detector.fit(X_train_scaled)

# 3. Predict OOD samples in the test sets
ood_preds_id = ood_detector.predict(X_test_scaled)
ood_preds_ood = ood_detector.predict(X_test_ood)

# Filter out detected OOD samples (-1 indicates an outlier/OOD)
X_test_id_filtered = X_test_scaled[ood_preds_id == 1]
y_test_id_filtered = y_test[ood_preds_id == 1]

X_test_ood_filtered = X_test_ood[ood_preds_ood == 1]
y_test_ood_filtered_indices = (ood_preds_ood == 1)
y_test_ood_filtered = y_test[y_test_ood_filtered_indices]


print(f"In-distribution: Detected and removed {np.sum(ood_preds_id == -1)}/{len(X_test_scaled)} samples as OOD.")
print(f"Out-of-distribution: Detected and removed {np.sum(ood_preds_ood == -1)}/{len(X_test_ood)} samples as OOD.")

# 4. Evaluate final model on both In-Distribution (ID) and filtered OOD sets
print("\n--- Final Model Evaluation ---")
y_pred_id = best_rf_model.predict(X_test_scaled)
y_pred_id_filtered = best_rf_model.predict(X_test_id_filtered)
y_pred_ood_filtered = best_rf_model.predict(X_test_ood_filtered)

print("\nPerformance on In-Distribution Test Set (before filtering):")
print(f"R^2: {r2_score(y_test, y_pred_id):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_id):.4f}")

print("\nPerformance on In-Distribution Test Set (after OOD filtering):")
print(f"R^2: {r2_score(y_test_id_filtered, y_pred_id_filtered):.4f}")
print(f"MAE: {mean_absolute_error(y_test_id_filtered, y_pred_id_filtered):.4f}")

print("\nPerformance on OOD Test Set (after OOD filtering):")
if len(y_test_ood_filtered) > 0:
    print(f"R^2: {r2_score(y_test_ood_filtered, y_pred_ood_filtered):.4f}")
    print(f"MAE: {mean_absolute_error(y_test_ood_filtered, y_pred_ood_filtered):.4f}")
else:
    print("No samples remained in the OOD set after filtering.")
    
# #### 4.4 Feature Importance Analysis

print("\n--- Feature Importance Analysis ---")
feature_importances = best_rf_model.feature_importances_
feature_names = X_train.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
print("Top 10 Feature Importances from the Tuned Random Forest Model:")
print(importance_df.head(10))

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
plt.title('Top 10 Feature Importances')
plt.show()


# #### Key Question 5: How can model results be applied to real-world air quality management?
# From the above Scenario-based Predictions, 
# air quality management should pay special attention to areas with high NOx emissions, as CO levels could also be elevated, posing potential health risks.
# This suggests that under high NOx and colder temperature conditions, CO levels tend to be elevated, which could be due to reduced atmospheric dispersion in cold weather, leading to higher pollution concentrations near the ground.

# In[ ]:




