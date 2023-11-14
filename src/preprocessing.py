#!/usr/bin/env python
# coding: utf-8

# # **Design Scorecards**
# ---

# ## **Pre-processing Training Set**

# In[1]:


#import library
import pandas as pd
import numpy as np
import sys
sys.path.append("../src")
#load configuration
import utils


# In[2]:


config_data = utils.config_load()
config_data


# In[10]:


def get_woe_map_dict():
    """
    Get the WOE mapping dictionary

    Returns
    -------
    dict : A dictionary containing the  mapping of characteristic, attribute, and their corresponding WOE values
    """
    #load the WOE table
    WOE_table = utils.pickle_load(config_data['WOE_table_path'])

    #initialize the dictionary
    WOE_map_dict = {}
    WOE_map_dict['Missing'] = {}
    
    #get unique characteristics
    unique_char = set(WOE_table['Characteristic'])
    for char in unique_char:
        #get the Attribute & WOE info for each characteristics
        current_data = (WOE_table
                            [WOE_table['Characteristic']==char]     
                            [['Attribute', 'WOE']])                
        
        #get the mapping
        WOE_map_dict[char] = {}
        for idx in current_data.index:
            attribute = current_data.loc[idx, 'Attribute']
            woe = current_data.loc[idx, 'WOE']

            if attribute == 'Missing':
                WOE_map_dict['Missing'][char] = woe
            else:
                WOE_map_dict[char][attribute] = woe
                WOE_map_dict['Missing'][char] = np.nan

    #validate data
    print('Number of key : ', len(WOE_map_dict.keys()))

    #dump
    utils.pickle_dump(WOE_map_dict, config_data['WOE_map_dict_path'])

    return WOE_map_dict


# In[11]:


WOE_map_dict = get_woe_map_dict()
WOE_map_dict


# In[12]:


def transform_woe(raw_data=None, type=None, config_data=None):
    """
    Replace data value with WOE scores
    
    Args
    ----
    raw_data : DataFrame
        Raw data to be transformed with WOE scores 
        If not provided, it is expected to load the data based on the specified type
    type : Str
        Type of data to transform, either "train" or "app"
        If provided, the raw data is loaded based on this type
    config_data : dict
        Configuration data including numeric columns and WOE map

    Returns
    -------
    pandas.DataFrame: Transformed data with WOE scores.

    This function replaces the original values in the raw data with WOE scores based on the provided WOE map. It takes care of both numerical and categorical columns.
    It is typically used for preparing data for credit scoring models.
    """
    #load the numerical columns
    numeric_cols = config_data['num_columns']

    #load the WOE_map_dict
    WOE_map_dict = utils.pickle_load(config_data['WOE_map_dict_path'])

    #load the saved data if type is not None
    if type is not None:
        raw_data = utils.pickle_load(config_data[f'{type}_path'][0])

    #map the data
    woe_data = raw_data.copy()
    for col in woe_data.columns:
        if col in numeric_cols:
            map_col = col + '_bin'
        else:
            map_col = col    

        woe_data[col] = woe_data[col].map(WOE_map_dict[map_col])

    #map the data if there is a missing value or out of range value
    for col in woe_data.columns:
        if col in numeric_cols:
            map_col = col + '_bin'
        else:
            map_col = col 

        woe_data[col] = woe_data[col].fillna(value=WOE_map_dict['Missing'][map_col])

    #validate
    print('Raw data shape : ', raw_data.shape)
    print('WOE data shape : ', woe_data.shape)

    #dump data
    if type is not None:
        utils.pickle_dump(woe_data, config_data[f'X_{type}_woe_path'])

    return woe_data


# In[13]:


#transform the train set
X_train_woe = transform_woe(type='train', config_data=config_data)
X_train_woe


# In[ ]:




