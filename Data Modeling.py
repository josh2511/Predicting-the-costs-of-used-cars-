#!/usr/bin/env python
# coding: utf-8

# # Predicting The Costs Of Used Cars

# The data science skills can help you predict the price of a used car based on a given set of features discussed below.
# 
# Size of training set: 6,019 records
# 
# Size of test set: 1,234 records
# 
# FEATURES:
# 
# Name: The brand and model of the car.
# 
# Location: The location in which the car is being sold or is available for purchase.
# 
# Year: The year or edition of the model.
# 
# Kilometers_Driven: The total kilometres driven in the car by the previous owner(s) in KM.
# 
# Fuel_Type: The type of fuel used by the car.
# 
# Transmission: The type of transmission used by the car.
# 
# Owner_Type: Whether the ownership is Firsthand, Second hand or other.
# 
# Mileage: The standard mileage offered by the car company in kmpl or km/kg
# 
# Engine: The displacement volume of the engine in cc.
# 
# Power: The maximum power of the engine in bhp.
# 
# Seats: The number of seats in the car.
# 
# New_Price: The price of a new car of the same model.
# 
# Price: The price of the used car in INR Lakhs.

# In[89]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax


# In[90]:


train = pd.read_csv("Data_Train#.csv")
test = pd.read_csv("Data_Test#.csv")


# In[91]:


train.head()


# In[92]:


test.head()


# In[93]:


train.shape


# In[94]:


train.columns


# In[95]:


train.info()


# In[96]:


test.info()


# In[97]:


train.dtypes


# # Data Cleaning of Power

# In[98]:


a=[]
for i in train['Power']:
    a.append(str(i).split(' ')[0])


# In[99]:


train['Power']=a


# In[100]:


train['Power']=train['Power'].replace('null',np.nan)


# In[101]:


train['Power']=train['Power'].fillna(train['Power'].median())


# In[102]:


train['Power'].isnull().sum()


# In[103]:


train['Power']=train['Power'].astype('float64')


# In[104]:


z=[]
for i in test['Power']:
    z.append(str(i).split(' ')[0])


# In[105]:


test['Power']=z


# In[106]:


test['Power']=test['Power'].replace('null',np.nan)


# In[107]:


test['Power']=test['Power'].fillna(test['Power'].median())


# In[108]:


test['Power'].isnull().sum()


# In[109]:


test['Power']=test['Power'].astype('float64')


# # Data cleaning of Engine

# In[110]:


train['Engine']=train['Engine'].replace('null',np.nan)


# In[111]:


train['Engine'].isnull().sum()


# In[112]:


train['Engine'] = train['Engine'].fillna(train['Engine'].mode()[0])


# In[113]:


train['Engine'] = train.Engine.str.replace('CC', '').astype(float)


# In[114]:


test['Engine']=test['Engine'].replace('null',np.nan)


# In[115]:


test['Engine'].isnull().sum()


# In[116]:


test['Engine'] = test['Engine'].fillna(test['Engine'].mode()[0])


# In[117]:


test['Engine'] = test.Engine.str.replace('CC', '').astype(float)


# # Data cleaning of Seat Colmun

# In[118]:


train.Seats.isnull().sum()


# In[119]:


train.Seats.value_counts()


# In[120]:


train.Seats.unique()


# In[121]:


train.Seats.describe()


# In[122]:


train['Seats']=train['Seats'].fillna(train['Seats'].mode()[0])


# In[123]:


test['Seats']=test['Seats'].fillna(test['Seats'].mode()[0])


# # Data cleaning of Mileage Colmun

# In[124]:


train.Mileage.isnull().sum()


# In[125]:


f=[]
for i in train['Mileage']:
    f.append (str(i).split(' ')[0])


# In[126]:


train['Mileage']=f


# In[127]:


train['Mileage'] = train.Mileage.astype(float)


# In[128]:


train['Mileage'] = train['Mileage'].fillna(train['Mileage'].median())


# In[129]:


e=[]
for i in test['Mileage']:
    e.append (str(i).split(' ')[0])


# In[130]:


test['Mileage']=e


# In[131]:


test['Mileage'] = test.Mileage.astype(float)


# In[132]:


test['Mileage'] = test['Mileage'].fillna(test['Mileage'].median())


# # Cleaning data of Name column

# In[133]:


names = list(train.Name)
brand = []
model = []
for i in range(len(names)):
   try:
       brand.append(names[i].split(" ")[0].strip())
       try:
           model.append(" ".join(names[i].split(" ")[1:]).strip())
       except:
           pass
   except:
       print("ERR ! - ", names[i], "@" , i)
train["Brand"] =  brand
train["Model"] = model
train.drop(labels = ['Name'], axis = 1, inplace = True)


# In[134]:


names = list(test.Name)
brand = []
model = []
for i in range(len(names)):
   try:
       brand.append(names[i].split(" ")[0].strip())
       try:
           model.append(" ".join(names[i].split(" ")[1:]).strip())
       except:
           pass
   except:
       print("ERR ! - ", names[i], "@" , i)
test["Brand"] =  brand
test["Model"] = model
test.drop(labels = ['Name'], axis = 1, inplace = True)


# In[135]:


train.drop(labels = ['New_Price'], axis = 1, inplace = True)
test.drop(labels = ['New_Price'], axis = 1, inplace = True)


# # Re-ordering Data-set

# In[136]:


train= train[['Brand', 'Model', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
      'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats', 'Price']]
test= test[['Brand', 'Model', 'Location', 'Year', 'Kilometers_Driven', 'Fuel_Type', 'Transmission',
      'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']]


# In[137]:


train.head()


# In[138]:


test.head()


# In[139]:


for i in train.columns:
    print("Unique values in", i, train[i].nunique())


# In[140]:


for i in test.columns:
    print("Unique values in", i, test[i].nunique())


# In[141]:


train.isnull().sum()


# In[142]:


test.isnull().sum()


# In[143]:


train.Brand.value_counts()


# In[144]:


train.Location.value_counts()


# In[145]:


train.Fuel_Type.value_counts()


# In[146]:


train.Transmission.value_counts()


# In[147]:


train.Owner_Type.value_counts()


# In[148]:


train.Seats.value_counts()


# In[149]:


train.head()


# In[150]:


df1 = train.copy() 
df1['Age_of_car'] = '2019'
df1['Age_of_car'] = pd.to_datetime(df1['Age_of_car'],format='%Y')
df1['Year'] = pd.to_datetime(df1['Year'],format='%Y')
df1['Age_of_car'] = (df1['Age_of_car'] - df1['Year']).dt.days
train['Age_of_car'] = df1['Age_of_car']

df2 = test.copy() 
df2['Age_of_car'] = '2019'
df2['Age_of_car'] = pd.to_datetime(df2['Age_of_car'],format='%Y')
df2['Year'] = pd.to_datetime(df2['Year'],format='%Y')
df2['Age_of_car'] = (df2['Age_of_car'] - df2['Year']).dt.days
test['Age_of_car'] = df2['Age_of_car']

del df1, df2
train.drop(['Year'], axis=1, inplace=True)
test.drop(['Year'], axis=1, inplace=True)


# In[151]:


market = {'Maruti' : 45.5,
'Hyundai'  :    11.5,
'Honda' :     4.5,
'Toyota'    :  3.5,
'Mercedes-Benz' :  2.4,
'Volkswagen':   3.6,
'Ford'    :   2.5,
'Mahindra' : 5.5,
'BMW'   :      1.6,
'Audi'    :  2,
'Tata'  :      5.5,
'Skoda'       : 2,
'Renault'  :   1.8,
'Chevrolet' :   1.5,
'Nissan':   1,
'Land'    : 1,
'Jaguar'      :1,
'Fiat'  :    0.6,
'Mitsubishi'  : 0.5,
'Mini':      0.5,
'Volvo'    :   0.5,
'Porsche' :  0.2,
'Jeep'  :    0.3,
'Datsun' :  0.3,
'Force'  :      0.1,
'ISUZU'  :    0.1,
'Smart'  :      0.1,
'Lamborghini'  :  0.1,
'Ambassador'  :   0.1,
'Isuzu' :  0.1,
'Bentley': 0.1
}
train['Market_Share'] = train['Brand'].map(market)
train['Market_Share']=train['Market_Share'].replace('null',np.nan)
train['Market_Share']=train['Market_Share'].fillna(train['Market_Share'].median())


test['Market_Share'] = test['Brand'].map(market)
test['Market_Share']=test['Market_Share'].replace('null',np.nan)
test['Market_Share']=test['Market_Share'].fillna(test['Market_Share'].median())


# In[152]:


sns.boxplot( y=train["Power"] )
plt.show()


# In[153]:


train.Power.describe()


# In[154]:


def get_power(Power):

    
    if (Power >=250  and Power < 600):
        return 'High Power'
    elif (Power >= 100 and Power < 250):
        return 'Medium Power'
    elif (Power >= 0 and Power < 100):
        return 'Low Power'
    
train['Car_Power'] = train['Power'].apply(get_power)
train['Car_Power']=train['Car_Power'].replace('null',np.nan)
train['Car_Power']=train['Car_Power'].fillna(train['Car_Power'].mode()[0])


test['Car_Power'] = test['Power'].apply(get_power) 
test['Car_Power']=test['Car_Power'].replace('null',np.nan)
test['Car_Power']=test['Car_Power'].fillna(test['Car_Power'].mode()[0])


# In[155]:


sns.boxplot( y=train["Engine"] )
plt.show()


# In[156]:


train.Engine.describe()


# In[157]:


def get_engine(Engine):

    
    if (Engine >=3000  and Engine < 6000):
        return 'High Engine'
    elif (Engine >= 1500 and Engine < 3000):
        return 'Medium Engine'
    elif (Engine >= 0 and Engine < 1500):
        return 'Low Engine'
    
train['Car_Engine'] = train['Engine'].apply(get_engine)   
test['Car_Engine'] = test['Engine'].apply(get_engine) 


# In[158]:


sns.boxplot( y=train["Mileage"] )
plt.show()


# In[159]:


train.Mileage.describe()


# In[160]:


def get_mileage(Mileage):

    
    if (Mileage >=24  and Mileage < 40):
        return 'Best Mileage'
    elif (Mileage >= 12 and Mileage < 24):
        return 'Economy Mileage'
    elif (Mileage >= 0 and Mileage < 12):
        return 'Least Mileage'
    
train['Car_Mileage'] = train['Mileage'].apply(get_mileage)   
test['Car_Mileage'] = test['Mileage'].apply(get_mileage) 


# In[161]:


def get_Kilometers(Kilometers_Driven):

    
    if (Kilometers_Driven >=300000  and Kilometers_Driven < 800000):
        return 'High Driven'
    elif (Kilometers_Driven >= 100000 and Kilometers_Driven < 300000):
        return 'Medium Driven'
    elif (Kilometers_Driven >= 40000 and Kilometers_Driven < 100000):
        return 'Minimum Driven'
    elif (Kilometers_Driven >= 1 and Kilometers_Driven < 40000):
        return 'Least Driven'
    
train['Car_Driven'] = train['Kilometers_Driven'].apply(get_Kilometers)
train['Car_Driven']=train['Car_Driven'].replace('null',np.nan)
train['Car_Driven']=train['Car_Driven'].fillna(train['Car_Driven'].mode()[0])


test['Car_Driven'] = test['Kilometers_Driven'].apply(get_Kilometers) 
test['Car_Driven']=test['Car_Driven'].replace('null',np.nan)
test['Car_Driven']=test['Car_Driven'].fillna(test['Car_Driven'].mode()[0])


# In[162]:


train.info()


# In[163]:


train.Brand.value_counts()


# In[164]:


train.Fuel_Type.value_counts()


# In[165]:


train.isnull().sum()


# In[166]:


test.info()


# In[167]:


test.isnull().sum()


# In[168]:


train_df = pd.get_dummies(train, columns=['Brand', 'Model', 'Location', 'Fuel_Type', 'Transmission',
      'Owner_Type', 'Mileage','Car_Mileage', 'Engine','Car_Engine', 'Power','Car_Power', 'Seats','Car_Driven'],drop_first=True)
test_df = pd.get_dummies(test, columns=['Brand', 'Model', 'Location', 'Fuel_Type', 'Transmission',
      'Owner_Type', 'Mileage','Car_Mileage', 'Engine','Car_Engine', 'Power','Car_Power', 'Seats','Car_Driven'],drop_first=True)


# In[169]:


X = train_df.drop(labels=['Price'], axis=1)
y = train_df['Price'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 2)


# In[170]:


x_train.shape, y_train.shape, x_test.shape, y_test.shape


# In[171]:



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[172]:


from math import sqrt 
from sklearn.metrics import mean_squared_log_error


# In[173]:


import lightgbm as lgb
train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test)



param = {'objective':'regression','num_leaves':80,
                              'learning_rate':0.1, 'n_estimators':100,
                              'max_bin' : 30, 'bagging_fraction' : 0.8,
                              'bagging_freq' : 9, 'feature_fraction' :0.129,
                              'feature_fraction_seed':9, 'bagging_seed':9,
                              'min_data_in_leaf' :3, 'min_sum_hessian_in_leaf' : 6, 'random_state':10}
lgbm = lgb.train(params=param,
                 verbose_eval=100,
                 train_set=train_data,
                 valid_sets=[test_data])

y_pred = lgbm.predict(x_test)
print('RMSLE:', sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred))))


# ## Save the model

# In[174]:


import pickle
# open a file, where you ant to store the data
file = open('car_price.pkl', 'wb')

# dump information to that file
pickle.dump(lgbm, file)


# In[175]:


model = open('car_price.pkl','rb')


# In[176]:


forest = pickle.load(model)

