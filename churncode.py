import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl
import matplotlib.pyplot as plt

#reading the csv file

df=pd.read_csv('data.csv')

#checking input data

print df.head()

print np.unique(df['TotalCharges'])

print df.dtypes

#isolate data

churn_result=df['Churn']

y=np.where(churn_result=='Yes',1,0);

#convert string to number

churnspace=df

churnspace=churnspace.replace(['Yes','No','No internet service','No phone service'],[1,0,0,0])
    
churnspace=churnspace.replace(['Bank transfer (automatic)','Credit card (automatic)','Electronic check','Mailed check'],[1,2,3,4])

churnspace=churnspace.replace(['DSL','Fiber optic','Male','Female'],[1,2,1,0])

churnspace=churnspace.replace(['Month-to-month','One year','Two year'],[1,2,3])

#checking converted data

print churnspace.head()

#convert object to numeric in dataframe

#churnspace = churnspace.apply(pd.to_numeric, errors='coerce')

print churnspace.dtypes

#checking nan values

print np.any(np.isnan(churnspace))


#Description of data

print churnspace.describe()

#standard deviation

print churnspace.std()

'''
#histogram

churnspace.hist()

pl.show()

'''

#creating dummy columns

dummy_pm=pd.get_dummies(churnspace['PaymentMethod'],prefix='PaymentMethod')

print dummy_pm.head()

dummy_is=pd.get_dummies(churnspace['InternetService'],prefix='InternetService')

print dummy_is.head()

dummy_c=pd.get_dummies(churnspace['Contract'],prefix='Contract')

print dummy_c.head()

#creating intercept data

coltokeep=["Churn","gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","PaperlessBilling","MonthlyCharges"]

data=churnspace[coltokeep]
print data.head()

data['intercept']=1.0


#performing the regression

train_cols = data.columns[1:]
# Index([gre, gpa, prestige_2, prestige_3, prestige_4], dtype=object)

logit = sm.Logit(data['Churn'], data[train_cols])

# fit the model
result = logit.fit()

print result.summary()

#confidence level

print result.conf_int()

#odds-ratio

print np.exp(result.params)

# odds ratios and 95% CI
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', '97.5%', 'OR']
print np.exp(conf)


#cartesian data


ten = np.linspace(data['tenure'].min(), data['tenure'].max(), 10)



print ten

mc = np.linspace(data['MonthlyCharges'].min(), data['MonthlyCharges'].max(), 10)

print mc
'''
tc = np.linspace(data['TotalCharges'].min(), data['TotalCharges'].max(), 10)

print tc
'''
a=[1,0]

def cartesian(arrays, out=None):

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]

    return out


combos=pd.DataFrame(cartesian([a,a,a,a,ten,a,a,a,a,a,a,a,a,a,mc,[1.]]))

combos.columns=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','MonthlyCharges','intercept']

tcols=['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','MonthlyCharges','intercept']

combos['churn_pred']=result.predict(combos[train_cols])

print combos.head()
