
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


""" Sampling transactions datafreame """

df_trans_cancel = transactions[transactions.is_cancel == 1]
df_trans_continue = transactions[transactions.is_cancel==0]
df_downsampled = resample(df_trans_continue,replace = False, n_samples= 856851)
df_trans_total = pd.concat([df_trans_cancel,df_downsampled])
del df_trans_cancel,df_trans_continue



# extracting user ids from transactions sample
members = np.unique(df_trans_total.msno)
sample_members = np.random.choice(members,size=2500 )

sample_logs = pd.DataFrame()
for chunk in user_logs :
    mem_df = chunk[chunk['msno'].isin(sample_members)]
    sample_logs = sample_logs.append(mem_df)
    print mem_df.shape

sample_logs['date'] = sample_logs.apply(lambda x : pd.to_datetime(x['date'],format = '%Y%m%d'), axis = 1)

df_trans_total['transaction_date'] = df_trans_total.apply(lambda x : pd.to_datetime(x['transaction_date'],format = '%Y%m%d'), axis = 1)

df_trans_total['membership_expire_date'] = df_trans_total.apply(lambda x : pd.to_datetime(x['membership_expire_date'],format = '%Y%m%d'), axis = 1)

df_trans_total = df_trans_total[df_trans_total['msno'].isin(sample_members)]

features = pd.DataFrame()
features = df_trans_total.apply(lambda x : get_avg(x), axis = 1 )

def get_avg(row) :
    start = row['transaction_date']
    end = row['membership_expire_date']
    df = sample_logs[(sample_logs['date'] > start) & (sample_logs['date'] < end) & (sample_logs['msno'] == row['msno'])]
    result = df.mean()
    #del result['Unnamed: 0']
    result['msno'] = row['msno']
    return result


kk = sample_logs[sample_logs['msno']=='BxEnycifHwsYJkWm6iLQZ54Yw4fHCmYK4KOYkPaalaw=']

kk_features = features[features['msno']=='BxEnycifHwsYJkWm6iLQZ54Yw4fHCmYK4KOYkPaalaw=']

kk_trans = df_trans_total[df_trans_total['msno']=='BxEnycifHwsYJkWm6iLQZ54Yw4fHCmYK4KOYkPaalaw=']

df = sample_logs[(sample_logs['date'] > start) & (sample_logs['date'] < end) & (sample_logs['msno'] == row['msno'])]