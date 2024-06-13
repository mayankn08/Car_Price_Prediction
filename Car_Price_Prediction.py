#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
import pandas as pd
import numpy as np
import warnings as ww
ww.simplefilter(action='ignore')


# ### Reding the data

# In[2]:


automobile = pd.read_csv('Automobile_data224.csv')
automobile.head()


# In[3]:


c=automobile['make'].unique()
c


# ### Getting the data types of the data set

# In[4]:


automobile.dtypes


# ## Cleaning of the data

# ### Find out if there are null values

# In[5]:


automobile.isna().sum()


# In[6]:


automobile.replace('?', np.nan, inplace = True)


# In[7]:


automobile['normalized-losses']=automobile['normalized-losses'].astype(float)
automobile['bore']=automobile['bore'].astype(float)
automobile['stroke']=automobile['stroke'].astype(float)
automobile['horsepower']=automobile['horsepower'].astype(float)
automobile['peak-rpm']=automobile['peak-rpm'].astype(float)
automobile['price']=automobile['price'].astype(float)


# In[8]:


automobile.isnull().sum()


# In[9]:


# Handling missing values
automobile["normalized-losses"].fillna(automobile["normalized-losses"].median(), inplace=True)
automobile["bore"].fillna(automobile["bore"].median(), inplace=True)
automobile["stroke"].fillna(automobile["stroke"].median(), inplace=True)
automobile["horsepower"].fillna(automobile["horsepower"].median(), inplace=True)
automobile["peak-rpm"].fillna(automobile["peak-rpm"].median(), inplace=True)
automobile["price"].fillna(automobile["price"].median(), inplace=True)
automobile["num-of-doors"].fillna(automobile["num-of-doors"].mode().iloc[0], inplace=True)

automobile.head(10)


# ## Statistical Summery

# In[10]:


automobile.describe()


# ## Univariate Analysis

# ### Import libraries

# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# ### Vehicle make frequency diagram

# In[12]:


top_makes = automobile['make'].value_counts().nlargest(10)

plt.figure(figsize=(15, 5))
top_makes.plot(kind='bar', color='skyblue')  

# Title and labels
plt.title("Top 10 Vehicle Makes by Count")
plt.xlabel('Make')
plt.ylabel('Number of Vehicles')

# Display the counts on top of the bars
for i, value in enumerate(top_makes):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom')

# Show the plot
plt.show()


# ### Insurance risk ratings Histogram

# In[13]:


plt.figure(figsize=(10, 6))
automobile['symboling'].hist(bins=6, color='skyblue', edgecolor='black') 


plt.title("Distribution of Insurance Risk Ratings")
plt.xlabel('Risk Rating')
plt.ylabel('Number of Vehicles')
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.show()


# ### Normalized losses histogram

# In[14]:


plt.figure(figsize=(10, 6))
automobile['normalized-losses'].hist(bins=5, color='skyblue', edgecolor='black')  

plt.title("Distribution of Normalized Losses in Vehicles")
plt.xlabel('Normalized Losses')
plt.ylabel('Number of Vehicles')
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.show()


# ### Fuel type bar chart

# In[15]:


plt.figure(figsize=(10, 6))
fuel_type_counts = automobile['fuel-type'].value_counts()
fuel_type_counts.plot(kind='bar', color='skyblue', edgecolor='black')  
plt.title("Distribution of Fuel Types in Vehicles")
plt.xlabel('Fuel Type')
plt.ylabel('Number of Vehicles')

for i, value in enumerate(fuel_type_counts):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# ### Engine type pie diagram

# In[16]:


plt.figure(figsize=(8, 8))
aspiration_counts = automobile['aspiration'].value_counts()
aspiration_counts.plot.pie(autopct='%.2f', colors=['skyblue', 'lightcoral'], explode=(0.1, 0), shadow=True)

plt.title("Distribution of Aspiration Types in Vehicles")
plt.xlabel('Aspiration Type')
plt.ylabel('')  
plt.show()


# ### Horse power histogram

# In[17]:


horsepower_filtered = automobile['horsepower'][
    np.abs(automobile['horsepower'] - automobile['horsepower'].mean()) <= (3 * automobile['horsepower'].std())]

plt.figure(figsize=(10, 6))
horsepower_filtered.hist(bins=5, color='skyblue', edgecolor='black') 


plt.title("Distribution of Horsepower (Outliers Removed)")
plt.xlabel('Horsepower')
plt.ylabel('Number of Vehicles')

plt.grid(axis='y', linestyle='--', alpha=0.7)


# ### Curb weight histogram

# In[18]:


plt.figure(figsize=(10, 6))
automobile['curb-weight'].hist(bins=5, color='skyblue', edgecolor='black')  # Adjust color and edgecolor as needed

plt.title("Distribution of Curb Weight in Vehicles")
plt.xlabel('Curb Weight')
plt.ylabel('Number of Vehicles')


plt


# ### Drive wheels bar chart

# In[19]:


plt.figure(figsize=(10, 6))
drive_wheels_counts = automobile['drive-wheels'].value_counts()
drive_wheels_counts.plot(kind='bar', color='grey', edgecolor='black')  # Adjust color and edgecolor as needed

plt.title("Distribution of Drive Wheels Types in Vehicles")
plt.xlabel('Drive Wheels Type')
plt.ylabel('Number of Vehicles')
for i, value in enumerate(drive_wheels_counts):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom')

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# ### Number of doors bar chart

# In[20]:


plt.figure(figsize=(10, 6))
doors_counts = automobile['num-of-doors'].value_counts()
doors_counts.plot(kind='bar', color='grey', edgecolor='black')  

plt.title("Distribution of Number of Doors in Vehicles")
plt.xlabel('Number of Doors')
plt.ylabel('Number of Vehicles')


for i, value in enumerate(doors_counts):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom')


plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# ## Bivariate Analysis

# # Heat Map
# 

# In[21]:


corr_matrix = automobile.corr()
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
plt.figure(figsize=(13, 7))
heatmap = sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=30)

plt.title("Correlation Matrix Heatmap")

plt.show()


# 
# 
# 
# 
# 
# 
# 

# ### Boxplot of Price and make
#     

# In[22]:


plt.rcParams['figure.figsize'] = (23, 10)

ax = sns.boxplot(x="make", y="price", data=automobile, palette="viridis")  # Adjust palette as needed
ax.set_title("Boxplot of Car Prices by Make")
ax.set_xlabel('Car Make')
ax.set_ylabel('Car Price')

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

plt.show()


# # Effect on price due to different factos 

# # Scatter plot of price and engine size
# 
# 

# In[23]:


plt.figure(figsize=(10, 6))
plt.scatter(automobile['price'], automobile['engine-size'], alpha=0.7, edgecolors='w', linewidth=0.5)

coefficients = np.polyfit(automobile['price'], automobile['engine-size'], 1)
poly_equation = np.poly1d(coefficients)
plt.plot(automobile['price'], poly_equation(automobile['price']), color='red', linewidth=2)
plt.title("Scatter Plot with Regression Line\nPrice vs. Engine Size")
plt.xlabel('Car Price')
plt.ylabel('Engine Size')
plt.show()


# # Scatter plot of price and wheel base

# In[24]:


plt.figure(figsize=(10, 6))
plt.scatter(automobile['price'], automobile['wheel-base'], alpha=0.7, edgecolors='w', linewidth=0.5)

coefficients = np.polyfit(automobile['price'], automobile['wheel-base'], 1)
poly_equation = np.poly1d(coefficients)
plt.plot(automobile['price'], poly_equation(automobile['price']), color='red', linewidth=2)

plt.title("Scatter Plot with Regression Line\nPrice vs. Wheel Base")
plt.xlabel('Car Price')
plt.ylabel('Wheel Base')
plt.show()


# # Scatter plot of price and horsepower

# In[25]:


plt.figure(figsize=(10, 6))
plt.scatter(automobile['price'], automobile['horsepower'], alpha=0.7, edgecolors='w', linewidth=0.5)

coefficients = np.polyfit(automobile['price'], automobile['horsepower'], 1)
poly_equation = np.poly1d(coefficients)
plt.plot(automobile['price'], poly_equation(automobile['price']), color='red', linewidth=2)

plt.title("Scatter Plot with Regression Line\nPrice vs. Horsepower")
plt.xlabel('Car Price')
plt.ylabel('Horsepower')

plt.show()


# 
# # Bar graph of price and horsepower

# In[26]:


plt.figure(figsize=(10, 6))
plt.scatter(automobile['price'], automobile['horsepower'], alpha=0.7, edgecolors='w', linewidth=0.5)

coefficients = np.polyfit(automobile['price'], automobile['horsepower'], 1)
poly_equation = np.poly1d(coefficients)
plt.plot(automobile['price'], poly_equation(automobile['price']), color='red', linewidth=2)

plt.title("Scatter Plot with Regression Line\nPrice vs. Horsepower")
plt.xlabel('Car Price')
plt.ylabel('Horsepower')

plt.show()


# # Bar graph of price and number of cylinders

# In[27]:


grouped_data = automobile.groupby('num-of-cylinders')['price'].agg(['mean', 'count']).reset_index()

plt.figure(figsize=(12, 6))
bars = plt.bar(grouped_data['num-of-cylinders'], grouped_data['mean'], alpha=0.7, color='skyblue', edgecolor='black')

for bar, count in zip(bars, grouped_data['count']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, count, ha='center', va='bottom')

plt.title("Average Price and Count by Number of Cylinders")
plt.xlabel('Number of Cylinders')
plt.ylabel('Average Price')

plt.show()


# # Bar graph of price and number of drive wheels

# In[28]:


grouped_data = automobile.groupby('drive-wheels')['price'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(grouped_data.index, grouped_data['mean'], alpha=0.7, color='skyblue', edgecolor='black')

for bar, count in zip(bars, grouped_data['count']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{count}', ha='center', va='bottom')

plt.title("Average Price and Count by Drive Wheels Type")
plt.xlabel('Drive Wheels Type')
plt.ylabel('Average Price')

plt.show()


# # Bar graph of price and number of body style

# In[29]:


grouped_data = automobile.groupby('body-style')['price'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(grouped_data.index, grouped_data['mean'], alpha=0.7, color='skyblue', edgecolor='black')

for bar, count in zip(bars, grouped_data['count']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{count}', ha='center', va='bottom')

plt.title("Average Price and Count by Body Style")
plt.xlabel('Body Style')
plt.ylabel('Average Price')

plt.show()


# # Bar graph of price and number of number of doors

# In[30]:


grouped_data = automobile.groupby('num-of-doors')['price'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(grouped_data.index, grouped_data['mean'], alpha=0.7, color='skyblue', edgecolor='black')

for bar, count in zip(bars, grouped_data['count']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{count}', ha='center', va='bottom')

plt.title("Average Price and Count by Number of Doors")
plt.xlabel('Number of Doors')
plt.ylabel('Average Price')

plt.show()


# # Bar graph of price and number of fuel type

# In[31]:


grouped_data = automobile.groupby('fuel-type')['price'].agg(['mean', 'count']).sort_values(by='mean', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(grouped_data.index, grouped_data['mean'], alpha=0.7, color='skyblue', edgecolor='black')

for bar, count in zip(bars, grouped_data['count']):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f'{count}', ha='center', va='bottom')

plt.title("Average Price and Count by Fuel Type")
plt.xlabel('Fuel Type')
plt.ylabel('Average Price')

plt.show()


# ### Drive wheels and City MPG bar chart

# In[34]:


average_city_mpg = automobile.groupby('drive-wheels')['city-mpg'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(average_city_mpg.index, average_city_mpg, color='peru', alpha=0.7, edgecolor='black')

for bar, value in zip(bars, average_city_mpg):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{value:.2f}', ha='center', va='bottom')

plt.title("Average City MPG by Drive Wheels")
plt.xlabel('Drive Wheels')
plt.ylabel('Average City MPG')

plt.show()


# ### Drive wheels and Highway MPG bar chart

# In[35]:


average_highway_mpg = automobile.groupby('drive-wheels')['highway-mpg'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(average_highway_mpg.index, average_highway_mpg, color='peru', alpha=0.7, edgecolor='black')

for bar, value in zip(bars, average_highway_mpg):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f'{value:.2f}', ha='center', va='bottom')

plt.title("Average Highway MPG by Drive Wheels")
plt.xlabel('Drive Wheels')
plt.ylabel('Average Highway MPG')

plt


# # PERFORMING FEATURE EXTRACTION,FEATURE SELECTION AND MODEL FITTING 

# In[11]:


from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor as XGB


# In[12]:


# Separate the target variable (e.g., 'price') and the features
X = automobile.drop(columns=['price'])
y = automobile['price']


# In[13]:


# Define categorical and numerical columns
categorical_columns = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']
numerical_columns = ['symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg']


# In[14]:


# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Standardize numerical data
    ('pca', PCA(n_components=5)) ]) # Apply PCA for feature extraction


# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))])  # One-hot encoding for categorical data


# In[15]:


# Fit and transform the categorical data using the categorical transformer
X_categorical_transformed = categorical_transformer.fit_transform(X[categorical_columns])

# Get the feature names after one-hot encoding
feature_names = categorical_transformer.named_steps['onehot'].get_feature_names_out(categorical_columns)

# Create a DataFrame to display the preprocessed categorical features
categorical_features_df = pd.DataFrame(X_categorical_transformed, columns=feature_names)

# Display the preprocessed categorical features
print(categorical_features_df)


# In[16]:


# Fit and transform the numerical data using the numerical transformer
X_numerical_transformed = numerical_transformer.fit_transform(X[numerical_columns])

# Get the feature names after PCA (in this case, the columns are not explicitly named)

numerical_feature_names = [f'PCA_{i}' for i in range(X_numerical_transformed.shape[1])]

# Create a DataFrame to display the preprocessed numerical features
numerical_features_df = pd.DataFrame(X_numerical_transformed, columns=numerical_feature_names)

# Display the preprocessed numerical features
print(numerical_features_df)


# In[17]:


# Bundle preprocessing for both numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)])
preprocessor


# ## Linear Regression 

# In[18]:


# Create the final pipeline with preprocessing feature selection and linear regression model
linear_regression_model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('feature_selector', SelectKBest(score_func=f_regression, k=10)),  # Adjust the number of features
                        ('regressor', LinearRegression())])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit the model
linear_regression_model.fit(X_train, y_train)
r2_score=linear_regression_model.score(X_test,y_test)
print(f"R-squared (R2) Score: {r2_score:.2f}")


# In[19]:


# Define the number of splits for ShuffleSplit
n_splits = 5 

# Create a ShuffleSplit object
shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)


cross_val_scores = cross_val_score(linear_regression_model, X, y, cv=shuffle_split, scoring='r2')  

for i, score in enumerate(cross_val_scores, 1):
    print(f"Fold {i}: R-squared (R2) Score: {score:.2f}")

print(f"\nMean R2 Score:", cross_val_scores.mean())  


# ## Random Forest

# In[20]:


Random_Forest_Regressor = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  ])


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit the model
Random_Forest_Regressor.fit(X_train, y_train)

# Evaluate the model
r2_score = Random_Forest_Regressor.score(X_test, y_test)
print(f"R-squared (R2) Score: {r2_score:.2f}")


# In[21]:


# Create a ShuffleSplit object
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Use cross_val_score to perform cross-validation with ShuffleSplit
cross_val_scores = cross_val_score(Random_Forest_Regressor, X_train, y_train, cv=shuffle_split, scoring='r2')  # Use 'r2' or other scoring metrics

for i, score in enumerate(cross_val_scores, 1):
    print(f"Fold {i}: R-squared (R2) Score: {score:.2f}")

print(f"\nMean R2 Score:", cross_val_scores.mean())


# ## Gradient Boosting

# In[22]:


from sklearn.metrics import r2_score as sklearn_r2_score 
Gradient_Boosting_Regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
df_GB=Gradient_Boosting_Regressor.fit(X_train, y_train)

# Make predictions
y_pred = df_GB.predict(X_test)

# Evaluate the model
r2_value_model = sklearn_r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2_value_model:.2f}")


# In[23]:


# Create a ShuffleSplit cross-validator
shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Perform cross-validation
cross_val_scores = cross_val_score(Gradient_Boosting_Regressor, X, y, cv=shuffle_split, scoring='r2')

# Display the R2 scores for each fold
for i, score in enumerate(cross_val_scores, 1):
    print(f"Fold {i}: R-squared (R2) Score: {score:.2f}")

# Calculate the mean R2 score across all folds
mean_r2_score = cross_val_scores.mean()
print(f"\nMean R-squared (R2) Score: {mean_r2_score}")


# ## XGBoost

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score as sklearn_r2_score
from xgboost import XGBRegressor
import pandas as pd

# Assuming you have defined 'preprocessor', 'X', and 'y' somewhere before this code

# Define the XGBoost Regressor pipeline
XGBoost_Regressor = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
XGBoost_Regressor.fit(X_train, y_train)

# Make predictions
y_pred = XGBoost_Regressor.predict(X_test)

# Evaluate the model
r2_value = sklearn_r2_score(y_test, y_pred)
print(f"R-squared (R2) Score: {r2_value:.2f}")


# In[25]:


shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Perform cross-validation
cross_val_scores = cross_val_score(XGBoost_Regressor, X, y, cv=shuffle_split, scoring='r2')

# Display the R2 scores for each fold
for i, score in enumerate(cross_val_scores, 1):
    print(f"Fold {i}: R-squared (R2) Score: {score:.2f}")

# Calculate the mean R2 score across all folds
mean_r2_score = cross_val_scores.mean()
print(f"\nMean R-squared (R2) Score: {mean_r2_score}")


# ## Hyperparameter tuning 

# In[27]:


from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pandas as pd


def find_best_model_using_gridsearchcv(x, y):
    # Define the pipeline with preprocessing and Gradient Boosting model
    model_gb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])

    model_xgb = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor())
    ])

    model_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    model_lr = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Define the hyperparameter grid for Gradient Boosting
    param_grid_gb = {
        'regressor__n_estimators': [50, 80, 100], 
        'regressor__learning_rate': [0.01, 0.1, 0.2],  
        'regressor__max_depth': [3, 5, 7],  
        'regressor__min_samples_split': [2, 5, 10], 
        'regressor__min_samples_leaf': [1, 2, 4],  
        'regressor__subsample': [0.8, 0.9, 1.0],  
        'regressor__max_features': ['sqrt', 'log2', None],
    }

    # Define the hyperparameter grid for XGBoost
    param_grid_xgb = {
        'regressor__n_estimators': [50, 80, 100],
        'regressor__learning_rate': [0.01, 0.1, 0.2],
        'regressor__max_depth': [3, 4, 5]
    }

    # Define the hyperparameter grid for Random Forest
    param_grid_rf = {
        'regressor__n_estimators': [50, 80, 100],
        'regressor__max_depth': [None, 10, 20],
    }

    # Define the hyperparameter grid for Linear Regression
    param_grid_lr = {
        'regressor__fit_intercept': [True, False],
        'regressor__copy_X': [True, False],
        'regressor__n_jobs': [None, 1, 2, 4],  # Adjust the values based on your system capabilities
        'regressor__positive': [False],  # 'positive' parameter is specific to scikit-learn version 0.24.0 and later
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    # Grid search for Gradient Boosting
    gs_gb = GridSearchCV(model_gb, param_grid_gb, cv=cv, return_train_score=False)
    gs_gb.fit(x, y)

    scores.append({
        'model': 'Gradient_Boosting_Regressor',
        'best_score': gs_gb.best_score_,
        'best_params': gs_gb.best_params_
    })

    # Grid search for XGBoost
    gs_xgb = GridSearchCV(model_xgb, param_grid_xgb, cv=cv, return_train_score=False)
    gs_xgb.fit(x, y)

    scores.append({
        'model': 'XGBoost_Regressor',
        'best_score': gs_xgb.best_score_,
        'best_params': gs_xgb.best_params_
    })

    # Grid search for Random Forest
    gs_rf = GridSearchCV(model_rf, param_grid_rf, cv=cv, return_train_score=False)
    gs_rf.fit(x, y)

    scores.append({
        'model': 'Random_Forest_Regressor',
        'best_score': gs_rf.best_score_,
        'best_params': gs_rf.best_params_
    })

    # Grid search for Linear Regression
    gs_lr = GridSearchCV(model_lr, param_grid_lr, cv=cv, return_train_score=False)
    gs_lr.fit(x, y)

    scores.append({
        'model': 'Linear_Regression',
        'best_score': gs_lr.best_score_,
        'best_params': gs_lr.best_params_
    })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])



find_best_model_using_gridsearchcv(X, y)


# In[28]:


result_df = find_best_model_using_gridsearchcv(X, y)

# Display the results
for index, row in result_df.iterrows():
    print(f"Model: {row['model']}")
    print(f"Best Score: {row['best_score']:.4f}")
    print("Best Parameters:")
    for param, value in row['best_params'].items():
        print(f"  {param}: {value}")
    print("\n")
 


# In[ ]:




