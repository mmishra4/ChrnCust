
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
os.chdir('C:\\Users\\Perceptive Analytics\\Documents\\data\\churn_data')
from collections import Counter
user_logs = pd.read_csv('user_logs.csv',chunksize=100)
members = pd.read_csv('members_v3.csv')
transactions = pd.read_csv('transactions.csv')
train = pd.read_csv('train.csv')



sam_mem = members.sample(n = 20000)
sam_mem = pd.read_csv('sample.csv')

sam_mem.to_csv('sample.csv')
mem_ids = np.unique(sam_mem['msno'])
sample_logs = pd.DataFrame()

for chunk in user_logs :
    print chunk.head()
    break
    mem_df = chunk[chunk['msno'].isin(mem_ids)]
    sample_logs = sample_logs.append(mem_df)
    print mem_df.shape

mem_ids2 = np.unique(sample_logs['msno'])
sample_logs.to_csv('sample_logs.csv')
sample_logs = pd.read_csv('sample_logs.csv')
train_sample = train[train['msno'].isin(mem_ids2)]
data = sample_logs.merge(train_sample, how = 'right' , on = 'msno')
""" 
np.unique(transactions.payment_plan_days)
len(np.unique(data.msno))
"""

sample_members = np.unique(data.msno)
transactions_sample = transactions[transactions.msno.isin(sample_members)]

#pd.to_datetime(transactions_sample.iloc[1]['transaction_date'],format = '%Y%m%d')

transactions_sample['transaction_date'] = transactions_sample.apply(lambda x : pd.to_datetime(x['transaction_date'],format = '%Y%m%d'), axis = 1)

transactions_sample['membership_expire_date'] = transactions_sample.apply(lambda x : pd.to_datetime(x['membership_expire_date'],format = '%Y%m%d'), axis = 1) 

sample_logs['date'] = sample_logs.apply(lambda x : pd.to_datetime(x['date'],format = '%Y%m%d'), axis = 1)

features = pd.DataFrame()
features['customer_id'] =
features = transactions_sample.apply(lambda x : get_avg(x), axis = 1 )

def get_avg(row) :
    start = row['transaction_date']
    end = row['membership_expire_date']
    df = sample_logs[(sample_logs['date'] > start) & (sample_logs['date'] < end) & (sample_logs['msno'] == row['msno'])]
    result = df.mean()
    del result['Unnamed: 0']
    result['msno'] = row['msno']
    return result

kk = members.apply(lambda x : len(np.unique(x)), axis = 0)
k = train.apply(lambda x : len(np.unique(x)), axis = 0)

#df = sample_logs[(sample_logs['date'] > start) & (sample_logs['date'] < end) & (sample_logs['msno'] == "4ByQQFjzA47U+lJ972beNun+W2FWM+uT3DtZi/WxT3c=")]

del features['date']
features.to_csv('features.csv')

features = pd.read_csv('features.csv')
features['is_cancel'] = list(transactions_sample['is_cancel'])
features.dropna(inplace= True)
features['encoded_msno'] = encoder.fit_transform(features['msno'])

msno_onehot = onehot.fit_transform(features['encoded_msno'].reshape(-1,1))
msno_onehot = pd.DataFrame(msno_onehot.toarray())
msno_onehot.index = features.index
final_df = pd.merge(features,msno_onehot ,left_index= True , right_index= True)
target = final_df['is_cancel']
final_df['is_cancel'] = target
del final_df['is_cancel']
del final_df['encoded_msno']
del final_df['msno']

x_train,x_test,y_train,y_test = train_test_split(final_df,target,test_size= 0.3)

model = LogisticRegression(C= 0.5)
model = RandomForestClassifier()
model.fit(x_train,y_train)
preds = model.predict(x_test)
print accuracy_score(y_test,preds)
print confusion_matrix(y_test,preds)

""" Upsampling """

df_cancel = final_df[final_df.is_cancel == 1]
df_continue = final_df[final_df.is_cancel == 0]


df_downsampled = resample(df_continue,replace = False, n_samples= 1500)
df_downsampled_total = pd.concat([df_cancel,df_downsampled])

target = df_downsampled_total['is_cancel']
del df_downsampled_total['is_cancel']
x_train,x_test,y_train,y_test = train_test_split(df_downsampled_total,target,test_size= 0.3 )
model = RandomForestClassifier(n_estimators=100 , max_features= 50 , max_depth= 100 , class_weight= {0:1 , 1:100 })

model.fit(x_train,y_train)
preds = model.predict(x_train)

print accuracy_score(y_train,preds)
print confusion_matrix(y_train,preds)

preds = model.predict(x_test)

print accuracy_score(y_test,preds)
print confusion_matrix(y_test,preds)