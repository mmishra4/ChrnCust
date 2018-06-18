
import operator
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
onehot =  OneHotEncoder()
from sklearn.preprocessing import LabelEncoder
from __future__ import division
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
#from fancyimpute import KNN
from statistics import  mode
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
os.chdir('C:\\Users\\Perceptive Analytics\\Documents\\data\\churn_data')

""" Reading the data """
user_logs = pd.read_csv('user_logs.csv',chunksize=100000)
members = pd.read_csv('members_v3.csv')
transactions = pd.read_csv('transactions.csv')
train = pd.read_csv('train.csv')

""" Obtaining balanced train sample  """
train_churn = train[train.is_churn == 1]
train_no_churn = train[train.is_churn == 0]
train_churn_sample = resample(train_churn,replace = False, n_samples= 5000)
train_no_churn_sample = resample(train_no_churn,replace = False, n_samples= 30000)
train_sample = pd.concat([train_churn_sample,train_no_churn_sample])
del train_churn,train_no_churn,train_churn_sample,train_no_churn_sample
train_sample.to_csv('train_sample.csv')
train_sample = pd.read_csv('train_sample.csv')

""" Obtaining members sample """
members_sample = members[members.msno.isin(train_sample.msno)]
del members
members_sample.to_csv('members_sample.csv')
members_sample = pd.read_csv('members_sample.csv')

""" Obtaining transactions sample """
transactions_sample = transactions[transactions.msno.isin(train_sample.msno)]
transactions_sample = transactions[transactions.msno.isin(train_sample.msno)]
del transactions
transactions_sample['membership_expire_date'] = transactions_sample.apply(lambda x : pd.to_datetime(x['membership_expire_date'],format = '%Y%m%d'), axis = 1)
transactions_sample['transaction_date'] = transactions_sample.apply(lambda x : pd.to_datetime(x['transaction_date'],format = '%Y%m%d'), axis = 1)

transactions_sample.to_csv('transactions_sample.csv')
transactions_sample= pd.read_csv('transactions_sample.csv')

""" Obtaining user logs sample """
user_logs_sample = pd.DataFrame()
for chunk in user_logs :
    mem_df = chunk[chunk['msno'].isin(train_sample.msno)]
    user_logs_sample = user_logs_sample.append(mem_df)
    print mem_df.shape

#user_logs_sample['date'] = user_logs_sample.apply(lambda x : str(x['date'])[-8:] , axis= 1)
user_logs_sample['date'] = user_logs_sample.apply(lambda x : pd.to_datetime(x['date'], format = '%Y%m%d'), axis = 1)
user_logs_sample.to_csv('user_logs_sample.csv',index= False)
user_logs_sample = pd.read_csv('user_logs_sample.csv')

""" Finding the common users"""

final_sample_users = list(set(np.unique(user_logs_sample.msno)) & set(np.unique(train_sample.msno)) & set(np.unique(transactions_sample.msno)) & set(np.unique(members_sample.msno)))

""" Finding the final sample"""

members_sample = members_sample[members_sample.msno.isin(final_sample_users)]
transactions_sample = transactions_sample[transactions_sample.msno.isin(final_sample_users)]
user_logs_sample = user_logs_sample[user_logs_sample.msno.isin(final_sample_users)]
train_sample = train_sample[train_sample.msno.isin(final_sample_users)]

""" Checking whether the common users are present again"""
print (len(set(np.unique(user_logs_sample.msno)) & set(np.unique(train_sample.msno)) & set(np.unique(transactions_sample.msno)) & set(np.unique(members_sample.msno))))

""" Writing to disk for backup"""
members_sample.to_csv('members_sample.csv',index = False)
transactions_sample.to_csv('transactions_sample.csv',index = False)
train_sample.to_csv('train_sample.csv',index = False)
user_logs_sample.to_csv('user_logs_sample.csv',index = False)
trans_static.to_csv('trans_static.csv',index= False)
zscores.to_csv('zscores.csv',index= False)

""" Reading from backup """
members_sample = pd.read_csv('members_sample.csv')
transactions_sample = pd.read_csv('transactions_sample.csv')
train_sample = pd.read_csv('train_sample.csv')
user_logs_sample = pd.read_csv('user_logs_sample.csv',parse_dates= ['date'])
trans_static = pd.read_csv('trans_static.csv')
trans_static = pd.read_csv('latest_trans_static.csv')
zscores = pd.read_csv('zscores.csv')

""" Functions required """

def get_most_frequent_payment(row):
    transactions_user = transactions_sample[transactions_sample['msno']==row['msno']]
    return int(Counter(transactions_user.payment_method_id).most_common(1)[0][0])

def get_most_frequent_plan(row):
    transactions_user = transactions_sample[transactions_sample['msno']==row['msno']]
    return int(Counter(transactions_user.payment_plan_days).most_common(1)[0][0])

def percent_auto_renew(row):
    transactions_user = transactions_sample[transactions_sample['msno']==row['msno']]
    return sum(transactions_user['is_auto_renew'])/len(transactions_user)

def upgraded_times(row) :
    transactions_user = transactions_sample[transactions_sample['msno']==row['msno']]
    upgrades = 0
    downgrades = 0
    consistent = 0
    if len(transactions_user) > 1 :
        num_changes = len(transactions_user)-1
        plan_days = list(transactions_user['payment_plan_days'])
        for i in range(num_changes):
            if plan_days[i+1] > plan_days[i] :
                upgrades +=1
            elif plan_days[i+1] < plan_days[i] :
                downgrades +=1
            else :
                consistent +=1
        return pd.Series(
                    data=[upgrades / (num_changes), downgrades / (num_changes), consistent / (num_changes)])
    else :
        return pd.Series(data = [0,1,0])


def counts_cancel_before_expiry(row) :
    transactions_user = transactions_sample[transactions_sample['msno']==row['msno']]
    cancel_trans = transactions_user[transactions_user.is_cancel==1]
    counts = len(cancel_trans)
    if counts == 0:
        return pd.Series(data = [0,0])
    else :
        days_before_cancellation = cancel_trans.apply(lambda x : (int((x['membership_expire_date']-x['transaction_date']).days)+1)/(x['payment_plan_days']+1), axis = 1)
        max_days_before = np.max(days_before_cancellation)
        return pd.Series(data = [counts,max_days_before])

cust_id = '47dzYFrpmp++CUVtJpStU7UpAUFRlpPdj8tJ+GpBN08='

def get_z_score(cust_id) :
    transactions_cust = transactions_sample[transactions_sample['msno']==cust_id]
    user_logs_cust = user_logs_sample[user_logs_sample['msno']== cust_id]
    end = max(transactions_cust['transaction_date'])
    user_logs_cust_current = user_logs_cust[user_logs_cust['date'] > end]
    if len(user_logs_cust_current)==0 :
        return pd.Series(data = [cust_id,5,5,5,5,5,5,5], index = ['msno','num_100','num_50','num_25','num_985','num_75','num_unq','total_secs'])
    user_logs_cust_historical = user_logs_cust[user_logs_cust['date'] <= end]
    if len(user_logs_cust_historical)==0 :
        return pd.Series(data=[cust_id, 1, 1, 1, 1, 1, 1, 1],index=['msno', 'num_100', 'num_50', 'num_25', 'num_985', 'num_75', 'num_unq', 'total_secs'])
    historical_means = user_logs_cust_historical.mean()
    historical_std = user_logs_cust_historical.std()
    current_means = user_logs_cust_current.mean()
    z_scores = (current_means-historical_means)/historical_std
    z_scores['msno'] = cust_id
    return z_scores


def get_last_login(cust_no) :
    user_logs_cust = user_logs_sample[user_logs_sample.msno==cust_no]
    days_since_last_login = int((last_day - max(user_logs_cust.date)).days)
    return days_since_last_login


""" Members static features creation """

train_features_df = pd.DataFrame()
train_features_df['customer_id'] = final_sample_users

""" Members dataframe bd cleaning """
index_outliers = (members_sample.bd > 90) | (members_sample.bd < 1)
members_sample.loc[index_outliers,'bd'] = 26
# average of top 3 occurences of age
members_sample.describe(include= 'object')

""" Members dataframe imputing gender """
members_sample.gender.fillna('Not_given',inplace = True)

""" Members dataframe registered via """
Counter(members_sample.registered_via)
# No changes required

""" transactions static features creation """


trans_static = pd.DataFrame()
trans_static['msno'] = final_sample_users
trans_static['frequent_payment_method'] = trans_static.apply(lambda x : get_most_frequent_payment(x),axis = 1)
trans_static['frequent_payment_plan'] = trans_static.apply(lambda x : get_most_frequent_plan(x),axis = 1)
trans_static['autorenew_percent'] = trans_static.apply(lambda x : percent_auto_renew(x),axis = 1)
trans_static[['upgrades','downgrades','consistent']] = trans_static.apply(lambda x : upgraded_times(x),axis = 1)
trans_static[['cancelled_counts','percent_time_left_when_cancelled']] = trans_static.apply(lambda x : counts_cancel_before_expiry(x),axis = 1)

""" Other static features """
trans_static = trans_static.merge(members_sample, how = 'left' ,on = 'msno')
last_day = pd.to_datetime(20170228,format= '%Y%m%d')
trans_static['associated_time'] = trans_static.apply(lambda x : int((last_day - pd.to_datetime(x['registration_init_time'],format = '%Y%m%d')).days),axis = 1)
""" Number of days since last login """

users_last_login = user_logs_sample.groupby('msno')['date'].aggregate(max).reset_index()
trans_static = trans_static.merge(users_last_login, how = 'left' , on= 'msno')
del users_last_login
trans_static['days_since_last_login'] = trans_static.apply(lambda x: int((last_day - x['date']).days),axis = 1)
del trans_static['registration_init_time']
del trans_static['date']
trans_static.to_csv('latest_trans_static.csv')

""" Number of cancellations"""
cancellations = transactions_sample.groupby('msno')['is_cancel'].aggregate(sum).reset_index()
trans_static = trans_static.merge(cancellations, how = 'left' , on= 'msno')
trans_static.rename(columns = {'is_cancel' : 'total_cancels'},inplace= True)
del cancellations
trans_static.to_csv('latest_trans_static.csv')

""" Total plan price etc"""
totals = transactions_sample.groupby('msno')['payment_plan_days','plan_list_price', 'actual_amount_paid'].aggregate(sum).reset_index()
totals.rename(columns = {'payment_plan_days' : 'total_plan_days','plan_list_price' :'total_plan_price', 'actual_amount_paid' :'total_Actual_amount_paid'},inplace= True)
totals.to_csv('totals.csv')
trans_static = trans_static.merge(totals, how = 'left' , on= 'msno')
del totals
trans_static.to_csv('latest_trans_static.csv')

""" Average number of 25, 50 etc per day """
avgs = user_logs_sample.groupby('msno')['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs'].aggregate(np.mean).reset_index()

avgs.rename(columns = {'num_25' : 'avg_25','num_50' : 'avg_50','num_75' :'avg_75','num_985' : 'avg_985','num_100':'avg_100','num_unq' : 'avg_unq' , 'total_secs' : 'avg_secs'} , inplace = True)
trans_static = trans_static.merge(avgs, how = 'left' , on= 'msno')
trans_static.to_csv('latest_trans_static.csv',index= False)

"""Putting z scores and trans stats together"""

zscores = trans_static.apply(lambda x : get_z_score(x['msno']),axis = 1)
zscores.rename(columns = {'num_25' : 'z_25','num_50' : 'z_50','num_75' :'z_75','num_985' : 'z_985','num_100':'z_100','num_unq' : 'z_unq' , 'total_secs' : 'z_secs'} , inplace = True)
trans_static = trans_static.merge(zscores, how = 'left' , on= 'msno')
trans_static.to_csv('latest_trans_static.csv', index= False)

trans_static['gender'] = LabelEncoder().fit_transform(trans_static['gender'])

cat_var_list =['frequent_payment_method','frequent_payment_plan','city','gender','registered_via']
num_var_list  = [item for item in list(trans_static.columns) if item not in cat_var_list]
num_var_list.remove('msno')
""" one hot encoding cat variable"""

enc = OneHotEncoder(sparse= False)
encode = pd.DataFrame(enc.fit_transform(trans_static.loc[:,cat_var_list]))
cat_features = []
for i, name in enumerate(cat_var_list):
    l = len(np.unique(trans_static[name]))
    for k in range(l):
        f_nm = str(name) + '@' + str(k + 1)
        cat_features.append(f_nm)
all_features = cat_features + num_var_list
""" scaling numerical variables """
#numeric = Imputer().fit_transform(trans_static.loc[:,num_var_list])
numeric = trans_static.loc[:,num_var_list].fillna(value= 0)
numeric = numeric.replace([np.inf, -np.inf], 10)
numeric  = pd.DataFrame(MinMaxScaler().fit_transform(numeric))


""" Train and test split """

final_df = pd.concat([encode,numeric] , axis= 1)
final_df['target'] = train_sample['is_churn']
final_df = final_df.dropna(axis =0, how = 'any')
target = final_df['target']
del final_df['target']

#final_df = Imputer().fit_transform(final_df)
"""Modelling"""

x_train,x_test,y_train,y_test = train_test_split(final_df,target,test_size= 0.25)

model = RandomForestClassifier(n_estimators=100, class_weight= { 0:1 , 1:3} ,max_features= 60 , min_samples_leaf= 10)
model.fit(x_train,y_train)
preds = model.predict(x_train)
train_prob = model.predict_proba(x_train)
train_prob = [item[1] for item in train_prob]
print "train loss is ",log_loss(y_train,train_prob ,normalize= True)
print "train accuracy" , accuracy_score(y_train,preds)
print "Train report is"
print classification_report(y_train,preds)
print confusion_matrix(y_train,preds)


preds = model.predict(x_test)
test_prob = model.predict_proba(x_test)
print "test loss is ",log_loss(y_test,test_prob)
print "test accuracy" , accuracy_score(y_test,preds)
print "Test report is"
print classification_report(y_test,preds)
print confusion_matrix(y_test,preds)


""" Code for grid search """
#'class_weight' : ({ 0:1 , 1:3}, { 0:1 , 1:2})
model2 = RandomForestClassifier(class_weight = {0:1 , 1:3} , n_estimators=100 , max_features= 60)

parameters = {'n_estimators' : (50,100), 'max_features' : (20,30,40,60) , 'min_samples_leaf' : (3,6,10,20) }

parameters = {'min_samples_leaf' : (3,6,10,20) }

ada_model = AdaBoostClassifier(model2,n_estimators= 15)
grid_model = GridSearchCV(model2, param_grid= parameters,cv = 5 , scoring= ['neg_log_loss','f1','accuracy'] , refit= 'neg_log_loss')

ada_model.fit(x_train,y_train)
grid_model.fit(x_train,y_train)
grid_model = ada_model
preds = grid_model.predict(x_train)
train_prob = grid_model.predict_proba(x_train)
train_prob = [item[1] for item in train_prob]
print "train loss is ",log_loss(y_train,train_prob ,normalize= True)
print "train accuracy" , accuracy_score(y_train,preds)
print "Train report is"
print classification_report(y_train,preds)

results = pd.DataFrame(grid_model.cv_results_)
results.to_csv('second_sample_results2.csv')
preds = grid_model.predict(x_test)
test_prob = grid_model.predict_proba(x_test)
print "test loss is ",log_loss(y_test,test_prob)
print "test accuracy" , accuracy_score(y_test,preds)
print "Test report is"
print classification_report(y_test,preds)




""" Validation"""

feature_imps = list(model.feature_importances_)
features = {}
for item in all_features:
    f = str(item).split('@')[0]
    if f in features.keys():
        features[f] = round((features[f] + feature_imps[all_features.index(item)]), 4)
    else:
        features[f] = round(feature_imps[all_features.index(item)], 4)

print(sorted(features.items(), key=operator.itemgetter(1), reverse=True))


feature_i = pd.DataFrame(sorted(features.items(), key=operator.itemgetter(1), reverse=True))
feature_i.to_csv('feature_imp_second_sample.csv')


feature_i.to_csv('second_sample_feature_importance_sorted.csv')

""" Draft  """



# Best so far

model = RandomForestClassifier(n_estimators=200, max_depth= 15 , class_weight= { 0:1 , 1:3})



#print confusion_matrix(y_test,preds)

#print accuracy_score(y_train,preds)
#print confusion_matrix(y_train,preds)

preds = model.predict(x_test)

model = RandomForestClassifier(n_estimators=200, max_depth= 15 , class_weight= { 0:1 , 1:3})

print accuracy_score(y_test,preds)
print confusion_matrix(y_test,preds)

from sklearn.metrics import classification_report
print classification_report(y_test,preds)

df = transactions_sample[(transactions_sample['transaction_date'] > start) & (transactions_sample['transaction_date'] < end) & (transactions_sample['msno'] == msno)]

del trans_static['Unnamed: 0.1']

int((transactions_sample.loc[1,'membership_expire_date']- transactions_sample.loc[1,'transaction_date']).days)\
msno = 'DqK23ZwuiWQKekqsS1QZdekyQaRgzzJxXE8OlUV6R3M='
transactions_user = transactions_sample[transactions_sample['msno'] == msno]

def get_avg(row) :
    start = row['transaction_date']
    end = row['membership_expire_date']
    df = user_logs_sample[(user_logs_sample['date'] > start) & (user_logs_sample['date'] < end) & (user_logs_sample['msno'] == row['msno'])]
    result = df.mean()
    del result['Unnamed: 0']
    result['msno'] = row['msno']
    return result

""" Estimating the song length """

df = user_logs_sample.sample(50)
df['song_length'] = df.apply(lambda x : x['total_secs']/ (0.25 * x['num_25'] + 0.5*x['num_50'] + 0.75*x['num_75'] + 0.985*x['num_985']+ x['num_100']),axis = 1)


custs = list(sample['msno'])
for p in [1,2,4,7,10] :
    cust_id = custs[p]
    transactions_cust = transactions_sample[transactions_sample['msno']==cust_id]
    transactions_cust = transactions_cust.sort_values(by = 'transaction_date')
    start = transactions_cust.iloc[0, :]['transaction_date']
    end = transactions_cust.iloc[(len(transactions_cust) - 1), :]['transaction_date']
    print start,end
cust_id = "dxdxExD3J1t+Wwn/PCPj3WTDr2djNuX2ycnhw1aUvoA="
sample = trans_static.head(20)
new = sample.apply(lambda x : get_z_score(x['msno'],1),axis = 1)

