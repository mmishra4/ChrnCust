

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
from sklearn.preprocessing import OneHotEncoder
onehot =  OneHotEncoder()
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from collections import Counter


os.chdir('C:\\Users\\Perceptive Analytics\\Documents\\data\\churn_data')

user_logs = pd.read_csv('user_logs.csv',chunksize=100000)
members = pd.read_csv('members_v3.csv')
transactions = pd.read_csv('transactions.csv')
train = pd.read_csv('train.csv')

transactions_2yrs = transactions[(transactions['transaction_date'] > 20150228 )]

transactions_2yrs.to_csv('transactions_2yrs.csv')

transactions_2yrs_without_free = transactions_2yrs[transactions_2yrs['actual_amount_paid']!=0]

transactions['transaction_date'] = transactions.apply(lambda x : pd.to_datetime(x['transaction_date'],format = '%Y%m%d'), axis = 1)

df_trans_total['membership_expire_date'] = df_trans_total.apply(lambda x : pd.to_datetime(x['membership_expire_date'],format = '%Y%m%d'), axis = 1)



counts_year_wise = {}

for i in range(14):
    year_low = (2004+i)*10000 + 100
    year_high = (2005+i)*10000 + 100
    transactions_1yr = len(transactions[transactions['transaction_date'] < year_low]) -  len(transactions[transactions['transaction_date'] >= year_high])
    counts_year_wise[(2004+i)] = len(transactions_1yr)
    print str(2004+i),"Count is", len(transactions_1yr)
    del transactions_1yr
