#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats.stats as stats
import warnings; warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score,roc_curve,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB


# Linear Regression Model
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.utils import shuffle


# # Reading the data into python

# In[2]:


train_df = pd.read_excel(r'C:\Users\reach\Dropbox\PC\Desktop\dynamic_pricing.xlsx', sheet_name = 'train_df')
test_df = pd.read_excel(r'C:\Users\reach\Dropbox\PC\Desktop\dynamic_pricing.xlsx', sheet_name = 'test_df')

print(f'Shape of the dataframe: {train_df.shape}')


# The data has one file "dynamic_pricing.xlsx" with two sheets"train_df and test_df". Train_df sheet contains 863 rows with 7 features/columns.

# # Data description

#    Product : The Different product names   
#    Product_Brand : The Product brand names       
#    Category : The 6 different categories of Auto parts       
#    Subcategory : The different sub categoried of Auto parts        
#    Item_Rating : The Ratings given for the products by customers       
#    Date : Date of Purchase of the product
#    Selling_Price : The selling price of the product  

# # Problem statement:

# Create a ML model which can predict the auto parts price of a car & Trucks¶
# 
# 
# Target Variable: Selling_Price
# 
# Predictors: Product, Product_Brand, Category, Subcategory, Item_Rating, Date etc.

# # Basic Data Exploration for the data frames - train & test 

# This step is performed to guage the overall data. The volume of data, the types of columns present in the data. Initial assessment of the data should be done to identify which columns are Quantitative, Categorical or Qualitative.
# 
# This step helps to start the column rejection process. You must look at each column carefully and ask, does this column affect the values of the Target variable? For example in this case study, you will ask, does this column affect the selling price of the auto parts? If the answer is a clear "No", then remove the column immediately from the data, otherwise keep the column for further analysis.
# 
# There are four commands which are used for Basic data exploration in Python
# 
# head() : This helps to see a few sample rows of the data
# info() : This provides the summarized information of the data
# describe() : This provides the descriptive statistical details of the data
# nunique(): This helps us to identify if a column is categorical or continuous

# In[3]:


train_df.info()
print("=="*30)
train_df.head()


# In[4]:


test_df.info()
print("=="*30)
test_df.head()


# In[5]:


train_df.describe().T


# In[6]:


train_df.columns


# In[7]:


# Finging unique values for each column
# TO understand which column is categorical and which one is Continuous
# Typically if the numer of unique values are < 50 then the variable is likely to be a category otherwise continuous
train_df.nunique()


# # Looking at the distribution of Target variable

# If target variable's distribution is too skewed then the predictive modeling will not be possible.
# Bell curve is desirable but slightly positive skew or negative skew is also fine
# When performing Regression, make sure the histogram looks like a bell curve or slight skewed version of it. Otherwise it   impacts the Machine Learning algorithms ability to learn all the scenarios.

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Creating Bar chart as the Target variable is Continuous
train_df['Selling_Price'].hist()


# The data distribution of the target variable is not satisfactory to proceed further. There are no sufficient number of rows for each type of values to learn from. So lets drill down further. 

# In[9]:


sns.set(style='whitegrid', palette='muted')
fig, ax = plt.subplots(1,2, figsize=(12,6))

sns.distplot(train_df['Selling_Price'], kde=True, ax=ax[0])
sns.scatterplot(x='Item_Rating', y='Selling_Price', data=train_df, marker='o', color='r', ax=ax[1])

plt.tight_layout()
plt.show()


# We need to apply the transformation method to make it normal. Here, we will try to use np.log1p method.

# In[10]:


# Transform the target variable
y_target = np.log1p(train_df['Selling_Price'])


# In[11]:


fig, axes = plt.subplots(1,2,figsize=(10,5))
sns.distplot(train_df['Selling_Price'], kde=True, ax=axes[0])
sns.distplot(y_target, kde=True, ax=axes[1])
axes[0].set_title("Skewed Y-Values")
axes[1].set_title("Normalized Y-Values")
plt.show()


# Now we can see the comparison of distribution of the target variable. After the transformation the target variable looks normalized and we can proceed for the model development.

# # Visual Exploratory Data Analysis
Categorical variables: Bar plot
Continuous variables: Histogram
# ### Visualize distribution of all the Categorical Predictor variables in the data using bar plots

# We can spot a categorical variable in the data by looking at the unique values in them. 

# ##### Categorical Predictors:

# 'Category', 'Subcategory'

# We use bar charts to see how the data is distributed for these categorical columns.

# In[12]:


# Plotting multiple bar charts at once for categorical variables
# Since there is no default function which can plot bar charts for multiple columns at once
# we are defining our own function for the same

def PlotBarCharts(inpData, colsToPlot):
    get_ipython().run_line_magic('matplotlib', 'inline')
    
    import matplotlib.pyplot as plt
    
    # Generating multiple subplots
    fig, subPlot=plt.subplots(nrows=1, ncols=len(colsToPlot), figsize=(40,10))
    fig.suptitle('Bar charts of: '+ str(colsToPlot))

    for colName, plotNumber in zip(colsToPlot, range(len(colsToPlot))):
        inpData.groupby(colName).size().plot(kind='bar',ax=subPlot[plotNumber])


# In[13]:


#####################################################################
# Calling the function
PlotBarCharts(inpData=train_df, colsToPlot=['Category', 'Subcategory'])


# # Bar Charts Interpretation

# These bar charts represent the frequencies of each category in the Y-axis and the category names in the X-axis.
# 
# In the ideal bar chart each category has comparable frequency. Hence, there are enough rows for each category in the data for the ML algorithm to learn.
# 
# If there is a column which shows too skewed distribution where there is only one dominant bar and the other categories are present in very low numbers. These kind of columns may not be very helpful in machine learning. We confirm this in the correlation analysis section and take a final call to select or reject the column.

# ##### Selected Categorical Variables:

# 'Category', 'Subcategory'- All these categories will be used in further analysis

# ### Visualize distribution of all the Continuous Predictor variables in the data using histograms

# Based on the Basic Data Exploration, One continuous predictor variables 'Item_Ratings'.

# Histograms shows us the data distribution for a single continuous variable.
# 
# The X-axis shows the range of values and Y-axis represent the number of values in that range. For example, in the below histogram of "Item_Ratings", there are around 371 rows.
# 
# The ideal outcome for histogram is a bell curve or slightly skewed bell curve. If there is too much skewness, then outlier treatment should be done and the column should be re-examined, if that also does not solve the problem then only reject the column.

# ##### Selected Continuous Variables:

# Item_Ratings

# In[14]:



# Plotting histograms of multiple columns together
train_df.hist([ 'Item_Rating'], figsize=(18,10))


# # Outlier treatment & Missing values treatment

# Outliers are extreme values in the data which are far away from most of the values. You can see them as the tails in the histogram.
# There are below two options to treat outliers in the data.
# 
# Option-1: Delete the outlier Records. Only if there are just few rows lost.
# Option-2: Impute the outlier values with a logical business value
#     
# In this data no prominent outliers are present, hence, not treating outlier in this section
# 
# Missing values are treated for each column separately.
# If a column has more than 30% data missing, then missing value treatment cannot be done. That column must be rejected because too much information is missing.
# 
# There are below options for treating missing values in data.
# 
# Delete the missing value rows if there are only few records
# 
# Impute the missing values with MEDIAN value for continuous variables
# 
# Impute the missing values with MODE value for categorical variables
# 
# Interpolate the values based on nearby values
# 
# Interpolate the values based on business logic

# In[15]:


# Finding how many missing values are there for each column
train_df.isnull().sum()


# No missing values in this data!!

# # Relationship exploration: Continuous Vs Continuous -- Scatter Charts

# When the Target variable is continuous and the predictor is also continuous, we can visualize the relationship between the two variables using scatter plot and measure the strength of relation using pearson's correlation value.

# In[16]:


ContinuousCols=['Item_Rating']

# Plotting scatter chart for each predictor vs the target variable
for predictor in ContinuousCols:
    train_df.plot.scatter(x=predictor, y='Selling_Price', figsize=(15,5), title=predictor+" VS "+ 'Selling_Price')


# ### Scatter charts interpretation

# ###### What should you look for in these scatter charts?

# Trend. You should try to see if there is a visible trend or not. There could be three scenarios

# 1. Increasing Trend: This means both variables are positively correlated. In simpler terms, they are directly proportional to each other, if one value increases, other also increases. This is good for ML!
# 
# 2. Decreasing Trend: This means both variables are negatively correlated. In simpler terms, they are inversely proportional to each other, if one value increases, other decreases. This is also good for ML!
# 
# 3. No Trend: You cannot see any clear increasing or decreasing trend. This means there is no correlation between the variables. Hence the predictor cannot be used for ML.
# 
# 
# Based on this chart you can get a good idea about the predictor, if it will be useful or not. You confirm this by looking at the correlation value.

# In[17]:


# Calculating correlation matrix
ContinuousCols=['Item_Rating', 'Selling_Price']

# Creating the correlation matrix
CorrelationData=train_df[ContinuousCols].corr()
CorrelationData


# # Relationship exploration: Categorical Vs Continuous -- Box Plots

# When the target variable is Continuous and the predictor variable is Categorical we analyze the relation using Boxplots and measure the strength of relation using Anova test

# In[18]:


# Box plots for Categorical Target Variable "Selling_Price" and continuous predictors
CategoricalColsList=['Category', 'Subcategory']

import matplotlib.pyplot as plt
fig, PlotCanvas=plt.subplots(nrows=1, ncols=len(CategoricalColsList), figsize=(18,5))

# Creating box plots for each continuous predictor against the Target Variable "price"
for PredictorCol , i in zip(CategoricalColsList, range(len(CategoricalColsList))):
    train_df.boxplot(column='Selling_Price', by=PredictorCol, figsize=(5,5), vert=True, ax=PlotCanvas[i])


# # Box-Plots interpretation

# ##### What should you look for in these box plots?

# These plots gives an idea about the data distribution of continuous predictor in the Y-axis for each of the category in the X-Axis.
# 
# If the distribution looks similar for each category(Boxes are in the same line), that means the the continuous variable has NO effect on the target variable. Hence, the variables are not correlated to each other.
# 
# On the other hand if the distribution is different for each category(the boxes are not in same line!). It hints that these variables might be correlated with price.
# 
# In this data, all the categorical predictors looks correlated with the Target variable.
# 
# We confirm this by looking at the results of ANOVA test below
# 
# 

# # Statistical Feature Selection (Categorical Vs Continuous) using ANOVA test

# Analysis of variance(ANOVA) is performed to check if there is any relationship between the given continuous and categorical variable

# 1. Assumption(H0): There is NO relation between the given variables (i.e. The average(mean) values of the numeric Target variable is same for all the groups in the categorical Predictor variable)
# 
# 2. ANOVA Test result: Probability of H0 being true

# In[19]:


# Defining a function to find the statistical relationship with all the categorical variables
def FunctionAnova(inpData, TargetVariable, CategoricalPredictorList):
    from scipy.stats import f_oneway

    # Creating an empty list of final selected predictors
    SelectedPredictors=[]
    
    print('##### ANOVA Results ##### \n')
    for predictor in CategoricalPredictorList:
        CategoryGroupLists=inpData.groupby(predictor)[TargetVariable].apply(list)
        AnovaResults = f_oneway(*CategoryGroupLists)
        
        # If the ANOVA P-Value is <0.05, that means we reject H0
        if (AnovaResults[1] < 0.05):
            print(predictor, 'is correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
            SelectedPredictors.append(predictor)
        else:
            print(predictor, 'is NOT correlated with', TargetVariable, '| P-Value:', AnovaResults[1])
    
    return(SelectedPredictors)


# In[20]:


# Calling the function to check which categorical variables are correlated with target
# Calling the function to check which categorical variables are correlated with target
CategoricalPredictorList=['Product', 'Product_Brand','Category', 'Subcategory']
FunctionAnova(inpData=train_df, 
              TargetVariable='Selling_Price', 
              CategoricalPredictorList=CategoricalPredictorList)


# # Merging the data frame- train_df & test_df

# In[21]:


# Merge train and test data
tempset = pd.concat([train_df, test_df], keys=[0,1])


# # Dropping the uncorrelated features and Date column

# In[22]:


tempset.drop(['Date', 'Product'], axis=1, inplace=True)


# # Selecting final predictors for Machine Learning

# Based on the above tests, selecting the final columns for machine learning

# In[23]:


SelectedColumns=[ 'Product_Brand','Category', 'Subcategory','Item_Rating']

# Selecting final columns
DataForML=tempset[SelectedColumns]
DataForML.head()


# # Data Pre-processing for Machine Learning

# List of steps performed on predictor variables before data can be used for machine learning
# 
# 1. Converting all other nominal categorical columns to numeric using pd.get_dummies()
# 2. Data Transformation (Optional): Standardization/Normalization/log/sqrt. Important if you are using distance based algorithms like KNN, or Neural Networks

# In this data there is no Binary nominal variable. So we will try to convert the Ordinal Categorical columns to numeric.

# # Converting the Ordinal Categorical variable to numeric using get_dummies()

# In[24]:


# Getting the categorical columns
cat_data = tempset.select_dtypes(include=['object'])

# One-hot encoding
X_encode = pd.get_dummies(tempset, columns=cat_data.columns)

# Getting back the Tran and Test data
X_train, X_enc_test = X_encode.xs(0), X_encode.xs(1)


# # Preparing the data for fitting the model

# In[25]:


# Prepare X and y for fitting the model
y = X_train['Selling_Price'].values
X = X_train.drop('Selling_Price', axis=1).values

X_test = X_enc_test.drop('Selling_Price', axis=1)
y_test = X_enc_test['Selling_Price']


# In[26]:


# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=428)


# In[27]:


### Sandardization of data ###
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Choose either standardization or Normalization
# On this data Min Max Normalization produced better results

# Choose between standardization and MinMAx normalization
#PredictorScaler=StandardScaler()
PredictorScaler=MinMaxScaler()

# Storing the fit object for later reference
PredictorScalerFit=PredictorScaler.fit(X)

# Generating the standardized values of X
X=PredictorScalerFit.transform(X)

# Split the data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[28]:


# Sanity check for the sampled data
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # Initializing the Linear Regression with Ridge regularizer

# In[29]:


ridge_cv = RidgeCV(normalize=True,cv=10,gcv_mode='svd',scoring='neg_mean_squared_error')

#Initializing Linear Regression algorithm with Ridge regularizer(K-fold with 10 folds)
ridge_reg = TransformedTargetRegressor(regressor= ridge_cv,
                                      func=np.log1p,
                                      inverse_func=np.expm1)


# In[30]:


ridge_reg.fit(X, y)

# Predict the test data
predictions = ridge_reg.predict(X_test)


# In[31]:


final_df = pd.DataFrame({'Selling_Price': predictions})

final_df['Selling_Price'] = final_df.apply(lambda x: round(x, 2))
final_df = pd.concat([test_df, final_df['Selling_Price']], axis=1)


# In[32]:


final_df.head(20)


# In[57]:


final_df.to_excel(r'C:\Users\reach\Dropbox\PC\Desktop\Predicted_results.xlsx', index=False)


# Finally we can see and compare the predicted selling price with actual selling price

# In[58]:


# model evaluation for testing set

mae = metrics.mean_absolute_error(y_test, predictions)
mse = metrics.mean_squared_error(y_test, predictions)
r2 = metrics.r2_score(y_test, predictions)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))


# In[33]:


from sklearn.linear_model import LinearRegression
RegModel = LinearRegression()

# Printing all the parameters of Linear regression
print(RegModel)

# Creating the model on Training Data
LREG=RegModel.fit(X_train,y_train)
prediction=LREG.predict(X_test)

# Taking the standardized values to original scale


from sklearn import metrics
# Measuring Goodness of fit in Training data
print('R2 Value:',metrics.r2_score(y_train, LREG.predict(X_train)))


# In[34]:


sns.regplot(y_test, prediction, scatter_kws={"color": "green"}, line_kws={"color": "blue"}) 


# In[35]:


# model evaluation for testing set

mae = metrics.mean_absolute_error(y_test, prediction)
mse = metrics.mean_squared_error(y_test, prediction)
r2 = metrics.r2_score(y_test, prediction)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))


# Mean absolute error is 1.16 which shows that our algorithm is not that accurate, but it can still make good predictions. 
# 
# The value of the mean squared error is 5.20 which shows that we have some outliers.
# 
# The R2  score is -1.16 and it shows that our model doesn’t fit data very well because it cannot explain all the variance. 

# In[40]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, y_train);


# In[42]:


# Use the forest's predict method on the test data
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[43]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[55]:


from xgboost import XGBRegressor

my_model = XGBRegressor(n_estimators=1000, learning_rate=0.5)
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(X_train, y_train, verbose=False)


# In[59]:



# make predictions
predictions1 = my_model.predict(X_test)

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(predictions1, y_test)))


# In[60]:


# model evaluation for testing set

mae = metrics.mean_absolute_error(y_test, predictions1)
mse = metrics.mean_squared_error(y_test, predictions1)
r2 = metrics.r2_score(y_test, predictions1)

print("The model performance for testing set")
print("--------------------------------------")
print('MAE is {}'.format(mae))
print('MSE is {}'.format(mse))
print('R2 score is {}'.format(r2))


# In[ ]:




