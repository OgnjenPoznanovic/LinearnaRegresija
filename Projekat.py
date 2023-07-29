# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 18:25:11 2023


@author: Ognjen Poznanovic IN45-2019
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

data = pd.read_excel('D:/faks/PO/Projekat/Concrete_Data.xls')


pd.set_option('max_columns', None)
df = pd.DataFrame(data)
print(df.head(5))


print("\n\n\n")
print("Broj uzoraka i broj obelezja")
print(df.shape)

print("\n\n\n")
print(df.dtypes)


print(data.describe())

print("\n\n\n")
df.rename(columns={"Cement (component 1)(kg in a m^3 mixture)": "Cement",
                     "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":"Blast Furnace Slag",
                     "Fly Ash (component 3)(kg in a m^3 mixture)":"Fly Ash",
                     "Water  (component 4)(kg in a m^3 mixture)":"Water",
                     "Superplasticizer (component 5)(kg in a m^3 mixture)":"Superplasticizer",
                     "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":"Coarse Aggregate",
                     "Concrete compressive strength(MPa, megapascals) ":"CCS",
                     "Fine Aggregate (component 7)(kg in a m^3 mixture)":"Fine Aggregate", 
                     "Age (day)":"Age"}, 
            inplace = True)


print(df.isnull().sum() / df.shape[0] * 100)

#Graficki prikaz raspodele podataka

#CEMENT
plt.figure()
plt.hist(data['Cement'], bins = 30, label='Cement')
plt.legend()
plt.show()

plt.figure()
plt.boxplot(data['Cement'])
plt.legend()
plt.show()


#BLAST FURNACE SLAG
plt.figure()
plt.hist(data['Blast Furnace Slag'], bins = 30, label='Blast Furnace Slag')
plt.legend()
plt.show()

plt.figure()
plt.boxplot(data['Blast Furnace Slag'])
plt.legend()
plt.show()


#FLY ASH
plt.figure()
plt.hist(data['Fly Ash'], bins = 30, label='Fly Ash')
plt.legend()
plt.show()

plt.figure()
plt.boxplot(data['Fly Ash'])
plt.legend()
plt.show()


#WATER
plt.figure()
plt.hist(data['Water'], bins = 30, label='Water')
plt.legend()
plt.show()

plt.figure()
plt.boxplot(data['Water'])
plt.legend()
plt.show()


#SUPERPLASTICIZER
plt.figure()
plt.hist(data['Superplasticizer'], bins = 30, label='Superplasticizer')
plt.legend()
plt.show()

plt.figure()
plt.boxplot(data['Superplasticizer'])
plt.legend()
plt.show()


#COARSE AGGREGATE
plt.figure()
plt.hist(data['Coarse Aggregate'], bins = 30, label='Coarse Aggregate')
plt.legend()
plt.show()

plt.figure()
plt.boxplot(data['Coarse Aggregate'])
plt.legend()
plt.show()


#FINE AGGREGATE
plt.figure()
plt.hist(data['Fine Aggregate'], bins = 30, label='Fine Aggregate')
plt.legend()
plt.show()

plt.figure()
plt.boxplot(data['Fine Aggregate'])
plt.legend()
plt.show()


#AGE
plt.figure()
plt.hist(data['Age'], bins = 30, label='Age')
plt.legend()
plt.show()

plt.figure()
plt.boxplot(data['Age'])
plt.legend()
plt.show()



#KORELACIJA
plt.figure()
corr_mat = data.corr()
sb.heatmap(corr_mat, annot=True)
plt.show()



#LINEARNA REGRESIJA
data_lr = data

x = data_lr.drop(['CCS'], axis=1).copy()
y = data_lr['CCS'].copy()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)






#standardizovani podaci
scaler = StandardScaler()
data_lr_s = data
x_s = data_lr_s.drop(['CCS'], axis=1).copy()
y_s = data_lr_s['CCS'].copy()

scaler.fit(x_s)
X_s = scaler.transform(x_s)


x_s_train, x_s_test, y_s_train, y_s_test = train_test_split(X_s, y_s, test_size=0.05, random_state=42)





def model_evaluation(y, y_predicted, N, d):
    mse = mean_squared_error(y_test, y_predicted) 
    mae = mean_absolute_error(y_test, y_predicted) 
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    r2_adj = 1-(1-r2)*(N-1)/(N-d-1)

  
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    print('R2 adjusted score: ', r2_adj)
    
    
    res=pd.concat([pd.DataFrame(y.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))






print("\n\n\n 1. MODEL")


first_regression_model = LinearRegression(fit_intercept=True)


first_regression_model.fit(x_train, y_train)


y_predicted = first_regression_model.predict(x_test)


model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])


plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)


print("\n\n\n 1.1 MODEL")


first_regression_model = LinearRegression(fit_intercept=False)


first_regression_model.fit(x_s_train, y_s_train)


y_s_predicted = first_regression_model.predict(x_s_test)


model_evaluation(y_s_test, y_s_predicted, x_s_train.shape[0], x_s_train.shape[1])


plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)




print("\n\n\n 2. MODEL")


x = data_lr.drop(['CCS'], axis=1).copy()
y = data_lr['CCS'].copy()
x_p_train, x_p_test, y_p_train, y_p_test = train_test_split(x, y, test_size=0.05, random_state=42)


poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
x_poly = poly.fit_transform(x_p_train)

model = LinearRegression()
model.fit(x_poly, y_p_train)


x_test_poly = poly.transform(x_p_test)
y_pred = model.predict(x_test_poly)


model_evaluation(y_p_test, y_pred, x_p_train.shape[0], x_p_train.shape[1])




print("\n\n\n 2.1 MODEL")

x_p = data_lr.drop(['CCS'], axis=1).copy()
y_p = data_lr['CCS'].copy()

poly = PolynomialFeatures(degree=3, interaction_only=False, include_bias=False)
X_p = poly.fit_transform(x_p)

scaler.fit(X_p)
X_p_s = scaler.transform(X_p)

x_p_s_train, x_p_s_test, y_p_s_train, y_p_s_test = train_test_split(X_p_s, y_p, test_size=0.05, random_state=42)



model = Ridge(alpha=0.3) 
model.fit(x_p_s_train, y_p_s_train)


y_s_pred = model.predict(x_p_s_test)


model_evaluation(y_p_s_test, y_s_pred, x_p_s_train.shape[0], x_p_s_train.shape[1])


print("\n\n\n 3 MODEL")


ridge_model = Ridge(alpha=0.3)


ridge_model.fit(x_poly, y_p_train)

y_predicted = ridge_model.predict(x_test_poly)

model_evaluation(y_p_test, y_predicted, x_p_train.shape[0], x_p_train.shape[1])


print("\n\n\n 3.1 MODEL")


ridge_model = Ridge(alpha=0.3)


ridge_model.fit(x_s_train, y_s_train)


y_predicted = ridge_model.predict(x_s_test)

model_evaluation(y_s_test, y_s_predicted, x_s_train.shape[0], x_s_train.shape[1])



print("\n\n\n 3.2  MODEL")


ridge_model = Ridge(alpha=0.5)


ridge_model.fit(x_train, y_train)

y_predicted = ridge_model.predict(x_test)

model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])


plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)


print("\n\n\n 3.3 MODEL")


lasso_model = Lasso(alpha=0.5)


lasso_model.fit(x_train, y_train)


y_predicted = lasso_model.predict(x_test)


model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])



plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.show()
print("koeficijenti: ", lasso_model.coef_)



print("\n\n\n 3.4 MODEL")


lasso_model = Lasso(alpha=0.3)


lasso_model.fit(x_poly, y_p_train)


y_predicted = lasso_model.predict(x_test_poly)


model_evaluation(y_p_test, y_predicted, x_p_train.shape[0], x_p_train.shape[1])



print("\n\n\n 3.5 MODEL")


lasso_model = Lasso(alpha=0.5)


lasso_model.fit(x_s_train, y_s_train)


y_predicted = lasso_model.predict(x_s_test)


model_evaluation(y_s_test, y_s_predicted, x_s_train.shape[0], x_s_train.shape[1])


df1 = pd.DataFrame(data)
print("\n\n\n")
df1.rename(columns={"Cement (component 1)(kg in a m^3 mixture)": "Cement",
                     "Blast Furnace Slag (component 2)(kg in a m^3 mixture)":"Blast Furnace Slag",
                     "Fly Ash (component 3)(kg in a m^3 mixture)":"Fly Ash",
                     "Water  (component 4)(kg in a m^3 mixture)":"Water",
                     "Superplasticizer (component 5)(kg in a m^3 mixture)":"Superplasticizer",
                     "Coarse Aggregate  (component 6)(kg in a m^3 mixture)":"Coarse Aggregate",
                     "Concrete compressive strength(MPa, megapascals) ":"CCS",
                     "Fine Aggregate (component 7)(kg in a m^3 mixture)":"Fine Aggregate", 
                     "Age (day)":"Age"}, 
            inplace = True)






x = df1.drop(['CCS'], axis=1).copy()
y = df1['CCS'].copy()


lr = LinearRegression()
sfs1 = sfs(lr, k_features=6, forward=True, verbose=2, scoring='neg_mean_squared_error')

sfs1 = sfs1.fit(x, y)
feat_names = list(sfs1.k_feature_names_)
print(feat_names)



df1 = df1.drop(['Coarse Aggregate'], axis=1)
df1 = df1.drop(['Fine Aggregate'], axis=1)

x = df1.drop(['CCS'], axis=1).copy()
y = df1['CCS'].copy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)



print("\n\n\n 1. MODEL")


first_regression_model = LinearRegression(fit_intercept=True)


first_regression_model.fit(x_train, y_train)

y_predicted = first_regression_model.predict(x_test)

model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])

plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)



print("\n\n\n 2. MODEL")

poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
x_poly = poly.fit_transform(x_train)

model = LinearRegression()
model.fit(x_poly, y_train)


x_test_poly = poly.transform(x_test)
y_pred = model.predict(x_test_poly)


model_evaluation(y_test, y_pred, x_train.shape[0], x_train.shape[1])



print("\n\n\n 3 MODEL")


ridge_model = Ridge(alpha=1)


ridge_model.fit(x_train, y_train)

y_predicted = ridge_model.predict(x_test)

model_evaluation(y_test, y_predicted, x_train.shape[0], x_train.shape[1])


plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)

