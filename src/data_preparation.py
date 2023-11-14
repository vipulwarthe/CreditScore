#!/usr/bin/env python
# coding: utf-8

# ## **Load Data**
# ---------
# 
# **These descriptions provide an explanation of each column in the dataset:**
# 
# - **person_age** : Age
# - **person_income** : Annual income
# - **person_home_ownership** : Type of home ownership [Rent, Mortgage, Own, Other]
# - **person_emp_length** : Employment length (in years)
# - **loan_intent** : Intent behind loan
# - **loan_grade** : Loan grade based on credit [A-G]
# - **loan_amnt** : Loan amount
# - **loan_int_rate** : Interest rate for the loan
# - **loan_status** : Loan status [0 is non default 1 is default]
# - **loan_percent_income** : Percentage of income
# - **cb_person_default_on_file** : Historical default [Y, N]
# - **cb_preson_cred_hist_length** : Credit history length

# In[29]:


#load library and configuration
import pandas as pd 
import sys

#append a specific path to the system path
sys.path.append("../src")


# In[30]:


#import the 'utils' module which contains utility functions
import utils


# In[31]:


#load configuration or data using 'config_load()' function from the 'utils' module
config_data = utils.config_load()
#display the loaded configuration data
config_data


# In[32]:


def read_data():
    """Load data and dump data"""

    # Load data
    data_path = config_data['raw_dataset_path']
    data = pd.read_csv(data_path)

    # Validate data shape
    print("Data shape       :", data.shape)

    # Pickle dumping (save the result)
    dump_path = config_data['dataset_path']
    utils.pickle_dump(data, dump_path)

    return data


# In[33]:


#load the dataset and display
data = read_data()
data.head()


# ### **Sample Splitting**
# ---

# - Split input & output data and dump them
# - Update the config file to contain
#     - The input & output data path
#     - The output variable name
#     - The input columns name

# In[34]:


# Define response variable
response_variable = 'loan_status'

# Check the proportion of response variable
data[response_variable].value_counts(normalize = True)


# The proportion of the response variable, `loan_status`, is not quite balanced (in a ratio of 78:22).
# 
# To get the same ratio in training and testing set, define a stratified splitting based on the response variable, `loan_status`.

# In[35]:


def splitting_data(data):
    """
    Split the dataset into predictor variables (X) and the response variable (y)

    Parameters
    ----------
    data : DataFrame
        The dataset containing both predictor and response variable

    Returns
    -------
    X : DataFrame
        Predictor variables (feature)
    y : Series
        Response variable

    This function takes a dataset and separate it into predictor variables (X) and response variable (y)
    It also saves the predictor variables and response variable to pickle files
    """

    #define response variable
    response_variable = config_data['response_variable']
    
    #extract the response variable (y) from dataset
    y = data[response_variable]

    #extract the predictor variables (X)
    X = data.drop(columns = [response_variable],
                  axis = 1)
    
    #display the shape of X and y 
    print('y shape :', y.shape)
    print('X shape :', X.shape)

    #save the predictor variable (X) to a pickle file
    dump_path_predictors = config_data['predictors_set_path']
    utils.pickle_dump(X, dump_path_predictors)

    #save the response variable (y) to a pickle file
    dump_path_response = config_data['response_set_path']    
    utils.pickle_dump(y, dump_path_response)
    
    return X, y


# In[36]:


X, y = splitting_data(data)


# Split training and testing from each predictors (X) and response variable (y)
# 
# - Set stratify = y for splitting the sample with stratify, based on the proportion of response y.
# - Set test_size = 0.2 for holding 20% of the sample as a testing set.
# - Set random_state = 42 for reproducibility.

# In[37]:


#import library 
from sklearn.model_selection import train_test_split


# Update the config file to have train & test data path and test size.

# In[38]:


config_data = utils.config_load()
config_data


# In[39]:


def split_train_test():
    """
    Split the dataset into training and testing

    Returns
    -------
    X_train : pd.DataFrame
        Training predictor variables
    X_test : pd.DataFrame
        Testing predictor variables
    y_train : pd.Series
        Training response variable
    y_test : pd.Series
        Testing response variable
    """
    
    #load the X and y
    X = utils.pickle_load(config_data['predictors_set_path'])
    y = utils.pickle_load(config_data['response_set_path'])

    #split the data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify = y,
                                                        test_size = config_data['test_size'],
                                                        random_state = 42)
    #validate splitting
    print('X_train shape :', X_train.shape)
    print('y_train shape :', y_train.shape)
    print('X_test shape  :', X_test.shape)
    print('y_test shape  :', y_test.shape)

    #dump data
    utils.pickle_dump(X_train, config_data['train_path'][0])
    utils.pickle_dump(y_train, config_data['train_path'][1])
    utils.pickle_dump(X_test, config_data['test_path'][0])
    utils.pickle_dump(y_test, config_data['test_path'][1])

    return X_train, X_test, y_train, y_test


# In[40]:


#check the function
X_train, X_test, y_train, y_test = split_train_test()


# Check proportion of response variable default in each training and testing set.

# In[41]:


#check proportion of target variable on data training
y_train.value_counts(normalize = True)


# In[42]:


#check proportion of target variable on data testing
y_test.value_counts(normalize = True)

