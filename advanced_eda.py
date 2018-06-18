import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
os.chdir('C:\\Users\\Perceptive Analytics\\Documents\\data\\churn_data\\13265_sample')
"""  Reading the sample data"""
members_sample = pd.read_csv('members_sample.csv')
transactions_sample = pd.read_csv('transactions_sample.csv')
train_sample = pd.read_csv('train_sample.csv')
user_logs_sample = pd.read_csv('user_logs_sample.csv',parse_dates= ['date'])
trans_static = pd.read_csv('latest_trans_static.csv')


user_logs_sample.head()
churn_train = train_sample[train_sample.is_churn == 1]
no_churn_train = train_sample[train_sample.is_churn == 0]

kk = user_logs_sample.loc[3,'msno']
single_user = user_logs_sample[user_logs_sample.msno == kk]
df= single_user[['date','num_25','num_100']]
df['ratio'] = df.num_25/df.num_100
df.set_index('date' , inplace = True)

plt.figure(figsize= (8,8))
#plt.plot(df.index,df.num_25 , kind = 'kde')
df.plot(figsize= (16,4))
plt.show()

last_day = pd.to_datetime(20150228,format= '%Y%m%d')
time_line =  pd.Series(pd.date_range(last_day, periods=732))

churn_users_100 = churn_train.sample(n = 100)
no_churn_users_100 = no_churn_train.sample(n = 100)

sample_cust = 'ToNwyrArhJ64q67QkwPOCcWMnQZoU+inQjJ/dy1N4v8='
single_user = user_logs_sample[user_logs_sample.msno == sample_cust]
df = single_user[['date','num_25']]
final_df = final_df.merge(df, how= 'left',left_index= True , right_on= 'date' )

""" VIZ of 20 churn users """
no_churn_df = pd.DataFrame(index=time_line)
j = 0
for i in list(no_churn_users_100['msno'])[1:20]:
    single_user = user_logs_sample[user_logs_sample.msno == i]
    single_user['ratio'] = single_user['num_50']/single_user['num_100']
    df = single_user[['date', 'num_100']]
    df.rename(columns={'num_100': 'cust_' + str(j)}, inplace=True)
    no_churn_df = no_churn_df.merge(df, how='left', left_index=True, right_on='date')
    no_churn_df.set_index('date' , inplace = True)
    j = j + 1
    no_churn_df = no_churn_df.fillna(value=0)


""" VIZ of 20 no churn users """
churn_df = pd.DataFrame(index=time_line)
j = 0
for i in list(churn_users_100['msno'])[1:20]:
    single_user = user_logs_sample[user_logs_sample.msno == i]
    single_user['ratio'] = single_user['num_50']/single_user['num_100']
    df = single_user[['date', 'num_100']]
    df.rename(columns={'num_100': 'cust_' + str(j)}, inplace=True)
    churn_df = churn_df.merge(df, how='left', left_index=True, right_on='date')
    churn_df.set_index('date' , inplace = True)
    j = j + 1
    churn_df = churn_df.fillna(value=0)


plt.subplot(211)
plt.title('NO CHURN- 100% songs played (20 users)')
#plt.ylabel('Ratio' , rotation = 'horizontal')
plt.plot(no_churn_df.index,no_churn_df.iloc[:,:-1])
plt.subplot(212)
#plt.ylabel('Ratio of 25% songs to 100% songs')
#plt.yticks(range(0,50,10))
plt.title('CHURN')
plt.plot(churn_df.index,churn_df.iloc[:,:-1])
plt.show()

""" Discounts analysis"""

transactions_sample = transactions_sample.merge(train_sample, on = 'msno' , how = 'left')
discount_df = transactions_sample[(transactions_sample.plan_list_price - transactions_sample.actual_amount_paid) > 0]
dis_users = np.unique(discount_df.msno)
all_discount_df = transactions_sample[transactions_sample.msno.isin(dis_users)]
#dis_train = train_sample[train_sample.msno.isin(dis_users)]
dis_users_counts = discount_df.groupby(['msno'])['is_churn'].agg(['count']).reset_index()
dis_users_counts.columns = ['msno','discounted_transactions']

#dis_users_counts = dis_users_counts.merge(dis_train, on = 'msno', how = 'left')

all_discount_df = all_discount_df.groupby(['msno'])['is_churn'].agg(['count']).reset_index()
all_discount_df.columns = ['msno','total_transactions']
all_discount_df['discounted_transactions'] = dis_users_counts.discounted_transactions
all_discount_df = all_discount_df.merge(dis_train,on = 'msno')
all_discount_df['no_discount_transactions'] = all_discount_df.total_transactions - all_discount_df.discounted_transactions
disc_analysis = pd.crosstab(all_discount_df['no_discount_transactions'],all_discount_df['is_churn'])
disc_analysis['ratio'] = (disc_analysis[1]/(disc_analysis[0]+disc_analysis[1]))*100

plt.plot(figsize =(16,8))
plt.hist(disc_analysis.ratio , bins = disc_analysis.index)
plt.imshow(disc_analysis.ratio)
plt.show(block=True)
plt.interactive(False)


""" Churn analysis downgrade wise"""

trans_static['is_churn'] = train_sample.is_churn
down_analysis = pd.crosstab(trans_static.downgrades,trans_static['is_churn'])
down_analysis['ratio'] = (down_analysis[1]/(down_analysis[0]+down_analysis[1]))*100
down_analysis['down_bins'] = pd.cut(down_analysis.index,bins = 5 , labels = ['<20','20-40','40-60','60-80','>80'])
down_analysis['down_bins'] = pd.cut(down_analysis.index,bins = 10)
down = down_analysis.groupby(['down_bins'])['ratio'].agg(['mean'])
fig = plt.figure(figsize =(16,8))
ax= fig.add_subplot(111)
plt.plot(down.index, down.mean)
plt.savefig('pp.png',dpi = 100)
plt.show()
plt.interactive(False)

"""Analysing consistent % """

cons_analysis = pd.crosstab(trans_static.consistent,trans_static['is_churn'])
cons_analysis['ratio'] = (cons_analysis[1]/(cons_analysis[0]+cons_analysis[1]))*100
cons_analysis['cons_bins'] = pd.cut(cons_analysis.index,bins = 10 , labels = ['<10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80-90','>90'])
cons = cons_analysis.groupby(['cons_bins'])['ratio'].agg(['mean'])

fig = plt.figure(figsize =(16,8))
ax= fig.add_subplot(111)
plt.plot(down.index, down.mean)
plt.savefig('pp.png',dpi = 100)
plt.show()
plt.interactive(False)


""" bd analysis """

bd_analysis = pd.crosstab(trans_static.bd,trans_static['is_churn'])
bd_analysis['ratio'] = (bd_analysis[1]/(bd_analysis[0]+bd_analysis[1]))*100
bd_analysis['age_bins'] = pd.cut(bd_analysis.index , bins= [0,12,25,35,45,65,80])
bd_analysis.reset_index(inplace = True)
bd = bd_analysis.groupby(['age_bins'])['ratio'].agg(['mean'])
trans_static['age_group'] = pd.cut(trans_static.bd , bins= [0,12,25,35,45,65,80])
counts = trans_static.groupby(['age_group'])['cancelled_counts'].agg(['count'])

""" Churn Analysis """

actual = transactions_sample.groupby(['msno'])['membership_expire_date'].agg(['max']).reset_index()
deadline = pd.to_datetime(20170129,format= '%Y%m%d')
actual['max'] = pd.to_datetime(actual['max'])
actual['is_churn'] = np.where(actual['max'] < deadline ,1,0)


""" """
user_logs_sample['total'] = user_logs_sample[['num_25','num_50','num_75','num_985','num_100']].sum(axis = 1)
user_logs_sample['repeat_songs'] = user_logs_sample.total - user_logs_sample.num_unq
user_logs_sample['repeat_percent'] = user_logs_sample['repeat_songs'] / user_logs_sample.total
repeat_stats = user_logs_sample.groupby('msno')['repeat_percent'].agg(['mean','std']).reset_index()
repeat_stats = repeat_stats.merge(train_sample,on = 'msno', how = 'left')
repeat_analysis = repeat_stats.groupby(['is_churn'])['mean','std'].agg(['mean'])

plt.figure(figsize=(10,8))
fig = plt.figure(figsize =(16,8))
ax= fig.add_subplot(111)
plt.scatter(repeat_stats['mean'], repeat_stats['std'])
plt.savefig('pp.png',dpi = 100)
plt.show()
plt.interactive(False)

user_logs_sample[['num_25','num_50','num_75','num_985','num_100']]= user_logs_sample[['num_25','num_50','num_75','num_985','num_100']] / user_logs_sample['total']
num_25_stats = user_logs_sample.groupby('msno')['num_25'].agg(['mean','std']).reset_index()
num_25_stats = num_25_stats.merge(train_sample,on = 'msno', how = 'left')
num_25_analysis = repeat_stats.groupby(['is_churn'])['mean','std'].agg(['mean','std'])

plt.figure(figsize=(10,8))
ax= fig.add_subplot(111)
plt.xlabel('mean of num 25 songs % played over all days' , fontsize = 14)
plt.ylabel('Standard deviation', fontsize = 14)
plt.scatter(num_25_analysis['mean'], num_25_analysis['std'] , c= num_25_analysis['is_churn'])
plt.show()

user_logs_sample['total'] = user_logs_sample[['num_25','num_50','num_75','num_985','num_100']].sum(axis = 1)

user_logs_sample[['num_25','num_50','num_75','num_985','num_100']]= user_logs_sample[['num_25','num_50','num_75','num_985','num_100']] / user_logs_sample['total']
num_25_stats = user_logs_sample.groupby('msno')['num_25'].agg(['mean','std']).reset_index()
num_25_stats = num_25_stats.merge(train_sample,on = 'msno', how = 'left')
num_25_analysis = repeat_stats.groupby(['is_churn'])['mean','std'].agg(['mean','std'])
plt.figure(figsize=(10,8))
ax= fig.add_subplot(111)
plt.xlabel('mean of num 25 songs % played over all days' , fontsize = 14)
plt.ylabel('Standard deviation', fontsize = 14)
plt.scatter(num_25_analysis['mean'], num_25_analysis['std'] , c= num_25_analysis['is_churn'])
plt.show()

""" time series analysis """

sample = user_logs_sample[user_logs_sample.msno == 'iX6Vo22yoopS+jsr6OOBA7j+1kqn8/siGv56NNCbTDo=']
X = sample['num_25'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

sample.set_index(['date'],inplace=True)
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(sample['num_25'], model='additive' , freq=30)
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)
result.plot()
plt.show()