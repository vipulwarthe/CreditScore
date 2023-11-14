#!/usr/bin/env python
# coding: utf-8

# # **Scaling**
# ---

# ## **Create Scorecard**

# In[1]:


#import library
import pandas as pd
import numpy as np

#import library for modeling
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

import sys
sys.path.append("../src")
#load configuration
import utils


# In[2]:


config_data = utils.config_load()
config_data


# In[3]:


def scaling():
    """
    Assign score points to each attribute based on the best model's output

    Returns
    -------
    pandas.DataFrame: A table containing the characteristics, WOE values, parameter estimates, and score points

    This function calculates score points for each attribute based on the best logistic regression model and the Weight of Evidence (WOE) values
    It uses predefined references like PDO (Points to Double the Odds) and offset to transform the logistic regression model output into score points
    The resulting score points are stored in a table and saved to a file. The table includes the characteristics, WOE values, parameter estimates, and score points for each attribute
    """

    #define the references: score, odds, pdo
    pdo = config_data['pdo']
    score = config_data['score_ref']
    odds = config_data['odds_ref']

    #load the best model
    best_model_path = config_data['best_model_path']
    best_model = utils.pickle_load(best_model_path)

    #load the WOE table
    WOE_table_path = config_data['WOE_table_path']
    WOE_table = utils.pickle_load(WOE_table_path)

    #load the best model's estimates table
    best_model_summary_path = config_data['best_model_summary_path']
    best_model_summary = utils.pickle_load(best_model_summary_path)

    #calculate Factor and Offset
    factor = pdo/np.log(2)
    offset = score-(factor*np.log(odds))

    print('===================================================')
    print(f"Odds of good of {odds}:1 at {score} points score.")
    print(f"{pdo} PDO (points to double the odds of good).")
    print(f"Offset = {offset:.2f}")
    print(f"Factor = {factor:.2f}")
    print('===================================================')

    #define n = number of characteristics
    n = best_model_summary.shape[0] - 1

    #define b0
    b0 = best_model.intercept_[0]

    print(f"n = {n}")
    print(f"b0 = {b0:.4f}")

    #adjust characteristic name in best_model_summary_table
    numeric_cols = config_data['num_columns']
    for col in best_model_summary['Characteristic']:

        if col in numeric_cols:
            bin_col = col + '_bin'
        else:
            bin_col = col

        best_model_summary.replace(col, bin_col, inplace = True) 

    #,erge tables to get beta/parameter estimate for each characteristic
    scorecards = pd.merge(left = WOE_table,
                          right = best_model_summary,
                          how = 'left',
                          on = ['Characteristic'])
    
    #define beta and WOE
    beta = scorecards['Estimate']
    WOE = scorecards['WOE']

    #calculate the score point for each attribute
    scorecards['Points'] = (offset/n) - factor*((b0/n) + (beta*WOE))
    scorecards['Points'] = scorecards['Points'].astype('int')

    #validate
    print('Scorecards table shape : ', scorecards.shape)
    
    #dump the scorecards
    scorecards_path = config_data['scorecards_path']
    utils.pickle_dump(scorecards, scorecards_path)

    return scorecards


# In[4]:


#check the function
scorecards = scaling()
scorecards


# In[5]:


#calculate the min and max points for each characteristic
grouped_char = scorecards.groupby('Characteristic')
grouped_points = grouped_char['Points'].agg(['min', 'max'])
grouped_points


# In[6]:


#calculate the min and max score from the scorecards
total_points = grouped_points.sum()
min_score = total_points['min']
max_score = total_points['max']

print(f"The lowest credit score = {min_score}")
print(f"The highest credit score = {max_score}")


# ## **Predict the Credit Score**

# In[7]:


def get_points_map_dict():
    """
    Get the Points mapping dictionary

    Returns:
    dict: A dictionary containing the points mapping for each attribute and characteristic

    This function generates a points mapping dictionary based on the scorecards table
    It iterates through the table, extracts the characteristics, attributes, and their corresponding points, and organizes them into a dictionary structure
    The resulting dictionary is then saved to a file
    """
    #load the Scorecards table
    scorecards = utils.pickle_load(config_data['scorecards_path'])

    #initialize the dictionary
    points_map_dict = {}
    points_map_dict['Missing'] = {}
    unique_char = set(scorecards['Characteristic'])
    for char in unique_char:
        #get the Attribute & WOE info for each characteristics
        current_data = (scorecards
                            [scorecards['Characteristic']==char]    
                            [['Attribute', 'Points']])              
        
        #get the mapping
        points_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            points = current_data.loc[idx, 'Points']

            if attribute == 'Missing':
                points_map_dict['Missing'][char] = points
            else:
                points_map_dict[char][attribute] = points
                points_map_dict['Missing'][char] = np.nan

    #validate data
    print('Number of key : ', len(points_map_dict.keys()))

    #dump
    utils.pickle_dump(points_map_dict, config_data['points_map_dict_path'])

    return points_map_dict


# In[8]:


#check the function
get_points_map_dict()


# In[9]:


def transform_points(raw_data=None, type=None, config_data=None):
    """
    Replace data value with points

    Args
    ----
    raw_data (DataFrame, optional): The raw data to be transformed. If None, the data is loaded based on the specified 'type'
    type (str, optional): The type of data to be transformed (e.g., 'train', 'test'). If None, 'raw_data' must be provided
    config_data (dict, optional): Configuration data containing file paths and settings

    Returns
    -------
    DataFrame: The transformed data with values replaced by points

    This function replaces the values in the input data with their corresponding points based on the 'points_map_dict'
    It handles both numeric and categorical columns, mapping them to their respective points
    Missing or out-of-range values are also mapped to points
    The transformed data is returned, and if a 'type' is specified, it is saved to a file
    """
    #lLoad the numerical columns
    numeric_cols = config_data['num_columns']

    #load the points_map_dict
    points_map_dict = utils.pickle_load(config_data['points_map_dict_path'])

    #load the saved data if type is not None
    if type is not None:
        raw_data = utils.pickle_load(config_data[f'{type}_path'][0])

    #map the data
    points_data = raw_data.copy()
    for col in points_data.columns:
        if col in numeric_cols:
            map_col = col + '_bin'
        else:
            map_col = col    

        points_data[col] = points_data[col].map(points_map_dict[map_col])

    #map the data if there is a missing value or out of range value
    for col in points_data.columns:
        if col in numeric_cols:
            map_col = col + '_bin'
        else:
            map_col = col 

        points_data[col] = points_data[col].fillna(value=points_map_dict['Missing'][map_col])

    #dump data
    if type is not None:
        utils.pickle_dump(points_data, config_data[f'X_{type}_points_path'])

    return points_data


# In[10]:


#check the function on the train set
X_train_points = transform_points(type='train', config_data=config_data)

X_train_points


# In[11]:


def predict_score(raw_data, config_data):
    """
    Predict the credit score for a given dataset.

    Args
    ----
    raw_data (DataFrame): The raw data for which to predict the credit score.
    config_data (dict): Configuration data containing file paths and settings.

    Returns
    -------
    int: The predicted credit score.

    This function takes raw data as input, transforms it into points using the 'transform_points' function, and calculates the credit score by summing the points for each row
    The cutoff score specified in the configuration is used to make a recommendation (APPROVE or REJECT), and the predicted score is saved to a file
    """
    
    points = transform_points(raw_data = raw_data, 
                              type = None, 
                              config_data = config_data)
    
    score = int(points.sum(axis=1))
    
    cutoff_score = config_data['cutoff_score']

    if score > cutoff_score:
        print("Recommendation : APPROVE")
    else:
        print("Recommendation : REJECT")

    utils.pickle_dump(score, config_data['score_path'])

    return score


# In[12]:


# Check the function with raw data input
tes_input = {
    'person_age_bin': 25,
    'person_income_bin': 60000,
    'person_emp_length_bin': 1,
    'loan_amnt_bin': 8000,
    'loan_int_rate_bin': 10,
    'loan_percent_income_bin': 0.25,
    'cb_person_cred_hist_length_bin': 2,
    'person_home_ownership': 'RENT',
    'loan_intent': 'MEDICAL',
    'cb_person_default_on_file': 'N'
}

tes = pd.DataFrame(tes_input, index=[0])

tes


# In[13]:


#predict the credit score
predict_score(raw_data=tes, config_data = config_data)

