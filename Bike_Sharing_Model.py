#!/usr/bin/env python
# coding: utf-8

# # BIKE SHARING LINEAR REGRESSION MODEL

# ## Reading and Understanding Data

# In[108]:


# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')


# In[52]:


# read the data
bike_data = pd.read_csv('/Users/vernica.ahuja/Downloads/Linear Regression/Bike Sharing Assignment - Linear Regression/day.csv')


# In[53]:


# Shape of the dataframe
bike_data.shape


# In[54]:


# Subset of dataset
bike_data.head()


# ### Cleaning data

# In[55]:


#check datatype and null values within the dataframe
bike_data.info()


# 1. Data has 730 Rows and 16 Columns
# 2. All columns are either Float or Integer except one as object type

# In[56]:


# Unique values within each column
bike_data.nunique()


# Drop Columns:
#     
#     1. instant - it is a serial no. column
#     
#     2. dteday - it denotes the date for which the data is present (all unique values)
# 
#     3. Casual & registered - these are subset of the target variable 'cnt' 
#                              and thus they will not be available when we have to predict the 
#                              users who will rent the bikes

# In[57]:


# Drop columns
bike_data.drop(['instant','dteday','casual','registered'],axis=1,inplace=True)


# In[58]:


bike_data.shape


# In[59]:


# Check duplicate rows within dataset
bike_data.drop_duplicates(subset=None, inplace=True)
print(bike_data.shape)


# There are no duplicate rows

# In[60]:


bike_data.describe()


# In[61]:


# Renaming levels within categorical variables
bike_data.season = bike_data.season.map({1:'spring', 2:'summer', 3:'fall', 4:'winter'})


# In[62]:


bike_data.season.unique()


# In[63]:


bike_data.mnth.unique()


# In[64]:


bike_data.mnth = bike_data.mnth.map({1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',
                                     9:'Sep',10:'Oct',11:'Nov',12:'Dec'})


# In[65]:


bike_data.mnth.unique()


# In[66]:


bike_data.weekday = bike_data.weekday.map({0:'sun',1:'mon',2:'tue',3:'wed',4:'thu',5:'fri',6:'sat'})


# In[67]:


bike_data.weathersit = bike_data.weathersit.map({1:'Clear',2:'Misty',3:'Light_snowrain',4:'Heavy_snowrain'})


# In[68]:


bike_data.head()


# In[69]:


bike_data.info()


# In[70]:


# visualising numeric variables`
sns.pairplot(bike_data)
plt.show()


# 1. temp & atemp look linearly related to the target variable cnt
# 2. temp & atemp are also correlated to each other

# In[71]:


bike_data.columns


# In[72]:


# visualising categorical variables
plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
sns.boxplot(x='season',y='cnt',data=bike_data)
plt.subplot(2,4,2)
sns.boxplot(x='yr',y='cnt',data=bike_data)
plt.subplot(2,4,3)
sns.boxplot(x='weekday',y='cnt',data=bike_data)
plt.subplot(2,4,4)
sns.boxplot(x='mnth',y='cnt',data=bike_data)
plt.subplot(2,4,5)
sns.boxplot(x='holiday',y='cnt',data=bike_data)
plt.subplot(2,4,6)
sns.boxplot(x='workingday',y='cnt',data=bike_data)
plt.subplot(2,4,7)
sns.boxplot(x='weathersit',y='cnt',data=bike_data)


# ### Insights:
# 
#     1. Season: Demand for rental bikes is highest in the Fall season
#     2. Yr : Demand has increased in 2019 
#     3. Weekday does not show a direct relation with demand of rental bikes
#     4. Demand increases from March - Aug and then decreases for the rest of the month
#     5. Misty and Clear Weather has higher demand as compared to Light snowrain
#     

# ## Preparing data for modelling

# Encoding:
#     
#     1. Converting binary variables to 1/0 which already exists like yr, holiday, weekday etc.
#     2. Other categorical vars to dummy vars

# In[74]:


# Create Dummy Variables
bike_data = pd.get_dummies(data=bike_data,columns=['season','weekday','mnth','weathersit'],drop_first=True)
bike_data.head()


# In[76]:


# Split data into train and test
df_train,df_test = train_test_split(bike_data, train_size=0.7, random_state = 100)


# In[78]:


print(df_train.shape)
print(df_test.shape)


# In[85]:


# Scaling the train & test dataset using min-max scaler
scaler = MinMaxScaler()


# In[82]:


df_train.columns


# In[87]:


# appyling scaler to numerical variables
scaling_vars  = ['temp','atemp','hum','windspeed','cnt']


# In[88]:


df_train[scaling_vars] = scaler.fit_transform(df_train[scaling_vars])


# In[89]:


df_train.head()


# In[99]:


# correlation of variables with target variable
corr = df_train.corr()
plt.figure(figsize = (25,25))
sns.heatmap(corr,annot=True)
plt.show()


# ## Feature selection

# In[100]:


# Splitting target variable and predictor variables
y_train = df_train.pop('cnt')
X_train = df_train


# In[118]:


# Recursive Feature Elimination

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=15)
rfe = rfe.fit(X_train, y_train)


# In[119]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[120]:


# selecting the RFE selected variables
col = X_train.columns[rfe.support_]
print(col)


# In[121]:


# rejected variables
X_train.columns[~rfe.support_]


# In[125]:


X_train_rfe = X_train[col]


# ## Building a model

# In[126]:


# Building 1st linear regression model

X_train_1 = sm.add_constant(X_train_rfe)
lr_1 = sm.OLS(y_train,X_train_1).fit()
print(lr_1.summary())


# In[148]:


# VIF
def calculateVIF(df):
    vif = pd.DataFrame()
    X = df
    vif['Features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    vif = vif.sort_values(by="VIF",ascending=False)
    return vif


# In[153]:


calculateVIF(X_train_rfe)


# In[154]:


# Drop variables that have high VIF values
X_train_new = X_train_rfe.drop(['hum'], axis = 1)


# In[155]:


X_train_new.columns


# In[158]:


# Building 2nd linear regression model

X_train_2 = sm.add_constant(X_train_new)
lr_2 = sm.OLS(y_train,X_train_2).fit()
print(lr_2.summary())


# In[159]:


# Again running VIF function
calculateVIF(X_train_new)


# In[160]:


X_train_new = X_train_new.drop(['temp'], axis = 1)


# In[161]:


# again calculating VIF
calculateVIF(X_train_new)


# In[162]:


# Building 3rd linear regression model

X_train_3 = sm.add_constant(X_train_new)
lr_3 = sm.OLS(y_train,X_train_3).fit()
print(lr_3.summary())


# In[164]:


# Drop variables with high P-value
X_train_new = X_train_new.drop(['mnth_Jul'], axis = 1)


# In[165]:


# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# In[166]:


# Building 4th linear regression model

X_train_4 = sm.add_constant(X_train_new)
lr_4 = sm.OLS(y_train,X_train_4).fit()
print(lr_4.summary())


# In[167]:


# Drop variables with high P-value
X_train_new = X_train_new.drop(['holiday'], axis = 1)


# In[169]:


# Building 5th linear regression model

X_train_5 = sm.add_constant(X_train_new)
lr_5 = sm.OLS(y_train,X_train_5).fit()
print(lr_5.summary())


# In[170]:


# Run the function to calculate VIF for the new model
calculateVIF(X_train_new)


# Insights:
# 
# Consider lr_5 as the final model as VIF for all variables is <5 and P-value of all variables is less than 0.05.
# 
# Prob(F-Statistic) is also close to 0.

# In[171]:


y_train_pred = lr_5.predict(X_train_5)


# ## Model Evaluation

# ### Residual Analysis of the train data

# In[174]:


#Plot the histogram of the error terms

fig = plt.figure()
sns.distplot((y_train - y_train_pred), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)
plt.xlabel('Errors', fontsize = 18)


# ## Making predictions on test data

# ### Applying scaling on test data

# In[177]:


df_test[scaling_vars] = scaler.transform(df_test[scaling_vars])


# ### dividing into X_test & y_test

# In[179]:


y_test = df_test.pop('cnt')
X_test = df_test


# In[184]:


cols = X_train_new.columns


# In[186]:


X_test = X_test[cols]


# In[188]:


X_test_lm = sm.add_constant(X_test)


# In[190]:


y_test_pred = lr_5.predict(X_test_lm)


# In[192]:


from sklearn.metrics import r2_score


# In[193]:


r2 = r2_score(y_test, y_test_pred)


# In[196]:


r2 = round(r2,3)
r2


# R2 is 0.742 which is close to the adjust R2 of the train set i.e. 0.774

# In[198]:


round(lr_5.params,4)


# Equation:
#     
#     cnt = 0.5332 + 0.2480 * yr + 0.0564 * workingday + (-0.1887) * windspeed + (-0.2581) * season_spring + 
#     
#          (-0.0394) * season_summer + (-0.0743) * season_winter + (0.0648) * weekday_sat + (-0.1033) * mnth_Jan 
#             
#          + (0.0715) * mnth_Sep + (-0.3023) * weathersit_Light_snowrain + (-0.0874) * weathersit_Misty

# In[201]:


# Plotting y_test  and y_pred  to understand the spread
fig = plt.figure()
plt.scatter(y_test,y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize=20)
plt.xlabel('y_test', fontsize=18)
plt.ylabel('y_test_pred', fontsize=16)


# In[207]:


# Adjust R^2 of the test data

adjusted_r2 = round(1-((1-r2)*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))),4)
print(adjusted_r2)


# ### Comparison of Test & Train set:
#     
#     R^2 for Train set:  0.779
#     R^2 for Test set:   0.742
#     Adj R^2 for Train set: 0.774
#     Adj R^2 for Test set:  0.728
