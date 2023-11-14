#!/usr/bin/env python
# coding: utf-8

# # **Data Exploration**
# ---

# ### **Exploratory Data Analysis (EDA)**
# 
# Data exploration is only performed on the training dataset.

# In[1]:


#import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

#add the project source directory to the system path for importing custom utilities
sys.path.append("../src")

#import custom utility function from 'utils' module
import utils

#add the project source directory to the system path for importing custom utilities
sys.path.append("../src")

#import custom utility function from 'utils' module
import utils


# In[2]:


config_data = utils.config_load()
config_data


# In[3]:


def concat_data(type):
    """
    Concat the input (X) and output (y) data and save the result
    
    Parameters
    ----------
    data_type : str
        A string indicating the type of data (train or test) 
    """
    #load input (X) and output (y) data
    X = utils.pickle_load(config_data[f'{type}_path'][0])
    y = utils.pickle_load(config_data[f'{type}_path'][1])
    
    #concatenate X and y
    data = pd.concat((X, y),
                     axis = 1)

    #display and validate data
    print(f'Data shape:', data.shape)

    #dump concatenated data
    utils.pickle_dump(data, config_data[f'data_{type}_path'])
   
    return data


# In[4]:


#check the function for train data
data_train = concat_data(type='train')
data_train.head()


# #### Check for Missing Values

# In[5]:


# Check for missing values
data_train.isna().sum()


# In[6]:


# Check for data type
data_train.info()


# ### **Check Correlation**

# In[7]:


#display the column names of the training_data DataFrame
data_train.columns


# In[8]:


#initialize numeric and categoric column
num_columns = config_data['num_columns']
cat_columns = config_data['cat_columns']


# In[9]:


# Plot scatter plots for numerical columns
sns.set(style="ticks")
sns.pairplot(data_train[num_columns], height=2)
plt.show()


# From the above chart, it can be observed that there is a relationship between the variables:
# - Age & Income: Income tends to increase with age.
# 
# - Age & Employment Length: Older individuals tend to have longer employment lengths.
# 
# - Income & Loan Amount: Individuals with higher income tend to borrow more money.
# 
# - Interest Rate & Percentage of Income for Loan: The interest rate may be higher for loans that represent a larger percentage of income.
# 
# - Credit History Length & Age: Credit history length generally increases with age.

# #### **Check Multicollinearity**
# 
# Next, calculate the Pearson correlation coefficient between numerical predictors.

# In[10]:


config_data = utils.config_load()
config_data


# In[11]:


# Calculate Pearson correlation coefficient from numerical predictors
data_train_corr = data_train[num_columns].corr(method = 'pearson')
data_train_corr


# In[12]:


# Plot the heatmap correlation
plt.figure(figsize=(10, 9))
sns.heatmap(data=data_train_corr,
            cmap="plasma",
            annot=True)
plt.show()


# We may have multicollinearity between `person_age` and `cb_person_cred_hist_length`.
# - We will perform model selection or
# - Exclude `cb_person_cred_hist_length`

# In[13]:


#identify categorical columns (object-type) in the training_data DataFrame
cat_cols = [col for col in data_train.columns if data_train[col].dtypes == 'O']

#loop through each categorical column and print its value counts
for col in cat_cols:
    print(data_train[col].value_counts(), "\n")


# Key Insights from Categorical Column Analysis:
# 
# - Home Ownership:
#     - Dominated by individuals in the "RENT" category.
#     - "OTHER" category has minimal representation.
# 
# - Loan Intent:
#     - "EDUCATION" is the most common loan intent.
#     - "HOMEIMPROVEMENT" has the lowest count.
# 
# - Default on File:
#     - Majority have no default on file ("N").
#     - Few individuals have a default on file ("Y").
# 
