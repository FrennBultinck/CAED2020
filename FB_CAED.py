#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:43:32 2021

@author: frennbultinck
"""
#https://www.analyticsvidhya.com/blog/2020/01/build-your-first-machine-learning-pipeline-using-scikit-learn/

#https://www.youtube.com/watch?v=R15LjD8aCzc 

#PCA: https://www.youtube.com/watch?v=Lsue2gEM9D0

#Principal component regression: https://www.statology.org/principal-components-regression-in-python/

################ IMPORT AND CLEAN ################



#standard imports
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import StandardScaler



#import data from csv file
df = pd.read_csv('/Users/frennbultinck/Desktop/prepared_data.csv')
df.head()
df.info()

#create the 'diff' columns: # if a column contains numbers and NaNs, pandas will default to float64
mood_diff = df[['mood_pre','mood_post']].mean(axis=1)
#STAI_diff = df[['STAI_pre','STAI_post']].mean(axis=1)

#add the diff columns to the dataset 
df['mood_diff'] = mood_diff
#df['STAI_diff'] = STAI_diff
df.head()

#reduce to the columns that I need
df = df[[
    'condition','subject','gender','age','mood_diff', 
    'STAI_pre_1_1','STAI_pre_1_2','STAI_pre_1_3','STAI_pre_1_4','STAI_pre_1_5',
          'STAI_pre_1_6','STAI_pre_1_7',
          'STAI_pre_2_1','STAI_pre_2_2','STAI_pre_2_3','STAI_pre_2_4','STAI_pre_2_5',
          'STAI_pre_2_6','STAI_pre_2_7',
          'STAI_pre_3_1','STAI_pre_3_2','STAI_pre_3_3','STAI_pre_3_4','STAI_pre_3_5',
          'STAI_pre_3_6',
          'moral_judgment']]
df.info()
df.head()

#check the data type for each column in a dataframe at once 
df.dtypes





################ BUILD A PROTOTYPE MODEL ################

##### 1. data exploration and preprocessing #####

##### a. encode the categorical variables



''' ML models cannot work with string (categorical) data,
    convert the cat variables into numeric types '''
#check the data type for each column in a dataframe at once 
df.dtypes
''' df.dtypes
Out[11]: 
condition          object
subject             int64
gender             object
age                 int64
mood_diff         float64
STAI_pre            int64
moral_judgment    float64
dtype: object'''
df.head()

#convert object to int
df['gender'].replace(['male','female'], [1,2], inplace=True) #male = 1, female = 2
df['condition'].replace(['control','stress'], [1,2], inplace=True) #control = 1, stress = 2
df.head(10)



##### b. split the data in testing_data and training_data (first split!)



'''create a training set by taking a sample with a fraction of 0.8 from the overall rows of the df. 
#random_state corresponds to the seed, for reproducibility. 
#can be done like this: training_data = df.sample(frac=0.8, random_state=25) '''
training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)
training_data.head()
testing_data.head()

#drop the training data from the testing set
print(f"No. of training samples: {training_data.shape[0]}")
print(f"No. of testing samples: {testing_data.shape[0]}")

#check for null values --> none
training_data.isna().sum()

#%%%

#########PCA########## (on training_data)

##### normalize/scale  data 

'''the continuous variables in the data often have different scales (range from 0-1
or from 0-100 e.g.) this is why we have to normalize the data in a way theat the range 
of the variables is almost similar '''

training_data.head()
features = ['condition','subject','gender','age','mood_diff', 
    'STAI_pre_1_1','STAI_pre_1_2','STAI_pre_1_3','STAI_pre_1_4','STAI_pre_1_5',
          'STAI_pre_1_6','STAI_pre_1_7',
          'STAI_pre_2_1','STAI_pre_2_2','STAI_pre_2_3','STAI_pre_2_4','STAI_pre_2_5',
          'STAI_pre_2_6','STAI_pre_2_7',
          'STAI_pre_3_1','STAI_pre_3_2','STAI_pre_3_3','STAI_pre_3_4','STAI_pre_3_5',
          'STAI_pre_3_6']


# Fit on training set only.

#scaled_data = preprocessing.scale(training_data.T)
scaled_data = StandardScaler().fit_transform(training_data.T)
#create PCA object
pca = PCA()
#calculate loading scores and variation each component accounts for
pca.fit(scaled_data)

#generate coordinates
pca_training_data = pca.transform(scaled_data)
#scree plot: howmany components needed? 
#calcul percentage of variation that each Princ.comp. accounts for
per_var = np.round(pca.explained_variance_ratio_*100, decimals=2)
#create labels for the scree plot
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

#howmany components = 26
pca.n_components_

#create plot with matplotlib
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

'''almost all the var is along the first component, 
so a 2-D graph, using PC1 and PC2 should do a good job representing
the original data'''

#create 6 principal components
pca = PCA(n_components=6)
Principal_components=pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(data = Principal_components, columns = ['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6'])
print(pca_df)

#scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()


#only hold on to the components with an eigenvalue bigger than 1
print(pca.explained_variance_) 



#to draw a plot, we'll first put the new coordinates created by pca.transform(scaled.data)
#into a nice matrix where the rows have sample labels and the columns have PCA labels
pca_df = pd.DataFrame(pca_training_data, 
                      index= list('abcdefghijklmnopqrstuvwxyz'),
                      columns=labels)

#draw a scatter plot with a title and axis labels
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('PCA graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))

#add sample names to the graph
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))

#display the graph
plt.show()

##look at loading scores
#create pandas "series" object with the loading scores in PC1
#PC = 0 because the PC's are 0 indexed
#loading_scores = pd.Series(pca.components_[0], index = training_data)
pca = PCA(n_components=2)
transformed_data = pca.fit(train_X).transform(train_X)
eigenValues = pca.explained_variance_ratio_
print(pca.components_)
print(eigenValues)

#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html 
pca.fit(train_X)
train_X_pca = pca.transform(train_X)
print("original shape:   ", train_X.shape)
print("transformed shape:", train_X_pca.shape)


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(train_X[:, 0], train_X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal');


#%%%

################ MODEL BUILDING ################



'''done with preprocessing, we start building machine learning models over the data
Linear regression and Random Forest regressor will be used to predict the moral judgment
to compare performance of the models, a validation (or test set) that holds 25% of the 
data points, while the train set has 75%'''

#separate the independent and target variable, make 'new datasets'
training_data
#only leave gender, age, mood_diff and STAI_pre in the train_X data
train_X = training_data.drop(columns=['moral_judgment'])
#only leave target 'moral judgment' in the train_Y data
train_Y = training_data['moral_judgment']

train_X.shape
train_Y.shape



#Split the data in training and validating sets (2nd split)
X_train, X_test, Y_train, Y_test = train_test_split(train_X, train_Y, test_size=0.25, random_state=0)
''' test_size=0.25: we split the dataset in 2 parts (training set, test set) and the ratio
of the test set compared to the dataset is 0.25 (38 to 150)

random _state = the seed for the random number generator. if blank or 0, the RandomState 
instance used by np.random will be used instead. '''

#shape of train and test splits
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
'''((112, 4), (38, 4), (112,), (38,))'''




#%%

#####LinearRegression Model
#define regression model 
modelLR = linear_model.LinearRegression()

#build training model 
modelLR.fit(X_train, Y_train)

#apply trained model to make prediction on the test set
Y_pred = modelLR.predict(X_test)

print('Coefficients', modelLR.coef_)
print('Intercept', modelLR.intercept_) 
print('Mean squared error (MSE): %.2f ' % mean_squared_error(Y_test, Y_pred))
print('Coefficient df determination (R^2): %.2f ' % r2_score(Y_test, Y_pred))

#print(diabetes.feature_names)
# = training_data

'''
Coefficients [ 0.37713642 -0.00102889  0.21146201  0.07897627 -0.00548909 -0.07796516
 -0.1507992  -0.38139069  0.21377036  0.16633679  0.30563028  0.20270907
 -0.25708424 -0.33120712  0.12950975  0.23584581  0.19912885 -0.50951102
 -0.23423208  0.4285794  -0.06283666  0.01513183  0.20980504 -0.02020556
 -0.35115924]

Intercept 4.696511782998105
Mean squared error (MSE): 1.29 
Coefficient df determination (R^2): -0.44 

equation for linear regression model: (met STAI_pre, niet alle features)
Y = 0.27052513*(condition) -0.00107966*(subject) + 0.3706932*(gender)
+ 0.05940237*(age) -0.00058563*(mood_diff) -0.06843635*(STAI_pre)
+ intercept of 4.490924300498608
'''



#%%
#####LinearRegression Model

# #create an object of the LinearRegression Model
# model_LR = LinearRegression()

# #fit the model with the training data
# model_LR.fit(train_x, train_y)

# #predict the target on train and test data 
# predict_train = model_LR.predict(train_x)
# predict_test = model_LR.predict(test_x)

# #root Mean Squared Error on train and test data
# print('RMSE on train data: ', mean_squared_error(train_y, predict_train)**(0.5))
# print('RMSE on test data: ', mean_squared_error(test_y, predict_test)**(0.5))
# '''
# RMSE on train data:  0.4033078009803911
# RMSE on test data:  0.8928456037684097 '''

#%%
#####RandomForestRegressor


#create an object for the RandomForestRegressor = define model
model_RFR = RandomForestRegressor(max_depth=10)

#fit the model with the training data = build training model
model_RFR.fit(X_train, Y_train)

#predict the target on train and test data  = apply to make prediction on the test set
predict_train = model_RFR.predict(X_train)
predict_test = model_RFR.predict(X_test)

#root Mean Squared Error on train and test data
print('RMSE on train data: ', mean_squared_error(Y_train, predict_train)**(0.5))
print('RMSE on test data: ', mean_squared_error(Y_test, predict_test)**(0.5))

#%%DecisionTreeRegressor

# df.head(5)
# class node():
#     def__init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
#         self.feature_index = feature_index
#         self.threshold= threshold
#         self.left = left
#         self.right = right
# #the data
# Y_test
# Y_pred

#making the scatterplot 
'''
sns.scatterplot(Y_test, Y_pred, marker = '+') #+ sign as marker
sns.scatterplot(Y_test, Y_pred, alpha = 0.5) #translucent dots (to see if dots are stacking)




# ################ FEATURE IDENTIFICATION TO PREDICT TARGET ################

# #plot the 7 most important features
plt.figure(figsize=(10,7))
feat_importances = pd.Series(model_RFR.feature_importances_, index = train_X.columns)
print(feat_importances)
'''



#%% Principal component Regression 
#https://www.statology.org/principal-components-regression-in-python/





#define predictor and response variables
training_data
#only leave gender, age, mood_diff and STAI_pre in the train_X data
X = training_data.drop(columns=['moral_judgment'])
#only leave target 'moral judgment' in the train_Y data
y = training_data['moral_judgment']
X.shape
y.shape

#scale predictor variables
#pca.fit_transform(scale(X)):#This tells Python that each of the predictor variables should be scaled to have a mean of 0 and a standard deviation of 1. This ensures that no predictor variable is overly influential in the model if it happens to be measured in different units.
pca = PCA()
X_reduced = pca.fit_transform(scale(X))

#define cross validation method
#RepeatedKFold: This tells Python to use k-fold cross-validation to evaluate the performance of the model. For this example we choose k = 10 folds, repeated 3 times.
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

regr = LinearRegression()
mse = []

# Calculate MSE with only the intercept
score = -1*model_selection.cross_val_score(regr,
           np.ones((len(X_reduced),1)), y, cv=cv,
           scoring='neg_mean_squared_error').mean()    
mse.append(score)

# Calculate MSE using cross-validation, adding one component at a time
for i in np.arange(1, 6):
    score = -1*model_selection.cross_val_score(regr,
               X_reduced[:,:i], y, cv=cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)
    
# Plot cross-validation results    
plt.plot(mse)
plt.xlabel('Number of Principal Components')
plt.ylabel('MSE')
plt.title('moral judgment')

''''
Weird results for the crossval !
from internet: 
From the plot we can see that the test MSE decreases by adding in two principal components, yet it begins to increase as we add more than two principal components.

Thus, the optimal model includes just the first two principal components.--> copy from website
'''


#calculate the percentage of variance in the response variable explained 
#by adding in each principal component to the model:
np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
'''([ 41.2 ,  49.97,  55.92,  61.37,  66.04,  70.12,  73.35,  76.39,
        79.22,  81.72,  84.  ,  86.08,  88.01,  89.55,  91.01,  92.33,
        93.56,  94.71,  95.76,  96.77,  97.62,  98.38,  99.06,  99.64,
       100.  ])'''

'''By using just the first principal component, we can explain 41.2% of the variation in the response variable.
    By adding in the second principal component, we can explain 49.97% of the variation in the response variable.
--> use 11 components, untill we reach 84% explained variance

'''


#split the original dataset into a training and testing set and 
#use the PCR model with two principal components to make predictions on the testing set.


#split the dataset into training (70%) and testing (30%) sets (2nd split)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

#scale the training and testing data
X_reduced_train = pca.fit_transform(scale(X_train))
X_reduced_test = pca.transform(scale(X_test))[:,:1]

#train PCR model on training data 
regr = LinearRegression()
regr.fit(X_reduced_train[:,:1], y_train)

#calculate RMSE
pred = regr.predict(X_reduced_test)
np.sqrt(mean_squared_error(y_test, pred))
'''
We can see that the test RMSE turns out to be 0.947122. 
This is the average deviation between the predicted value for moral_judgment
 and the observed value for moral judgment for the observations in the testing set.








