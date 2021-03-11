#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_preprocessed = pd.read_csv('C:/Users/joe_h/Downloads/Absenteeism_data_preprocessed.csv')


# In[3]:


data_preprocessed.head()


# In[4]:


# great thing about regressions is that the model will give us an indication of which variables are important for the analysis


# ## Create the Targets

# In[5]:


# create 2 classes, 1 for people who were moderately absent, and another for people excessively absent
# take the median value of the 'Absenteeism Time in Hours' and use it as a cut-off line
# anything below median will be moderate, anything above will be excessive
data_preprocessed['Absenteeism Time in Hours'].median()


# In[6]:


# therefore, anyone absent for <3 hours will be moderately, and anyone absent for >3 hours will be excessive
# we'll place anyone <3 hours absent as 0 and anyone >3 hours absent as 1


# In[7]:


targets = np.where(data_preprocessed['Absenteeism Time in Hours'] >
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)
# arguments within function 'where' follow this generality: where(condition, value if True, value if False)
# this tells us whether condition is satisfied or not


# In[8]:


targets


# In[9]:


data_preprocessed['Excessive Absenteeism'] = targets


# In[10]:


data_preprocessed.head()


# ### Comment on the targets

# In[11]:


# using the median as a cut-off line is numerically stable and rigid
# we have implicity balanced the dataset

# we now sum up all values of all the targets and divide/ by the total number of the targets (aka the shape on axis [0])
targets.sum() / targets.shape[0]


# In[12]:


# around 46% of the targets are 1s, so 54% of the targets are 0s
# this balance is sufficient for this exercise (between 45-55%)


# In[13]:


# since we have extracted the data from the target column 'Absenteeism Time in Hours', can now drop it from the dataframe
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours', 'Day of the Week',
                                            'Daily Work Load Average', 'Distance to Work'], axis=1)
# after completing this entire notebook analysis, I have come back to remove unimportant variables,
# these were found to be day of the week, daily work load and distance to work


# In[14]:


# lets check if the 2 variables are the same object (using 'is')
data_with_targets is data_preprocessed


# In[15]:


# Nice, since it is false, we have successfully droppped the column from the dataframe
data_with_targets.head()


# ## Select the Inputs for the regression

# In[16]:


data_with_targets.shape


# In[17]:


# the iloc method is used to select slices of data by position when specified rows and columns wanted
# we want to select all columns except the last one (Excessive Absenteeism)

data_with_targets.iloc[:,:-1]
# -1 as this begins count from the end
# as there are 15 columns in total and the one we don't want to include is the 15th (at the end) which is (-1)


# In[18]:


unscaled_inputs = data_with_targets.iloc[:,:-1]


# ## Standardise the Data

# In[19]:


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

class CustomScaler(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns):
                 self.scaler = StandardScaler()
                 self.columns = columns
                 self.mean = None
                 self.var_ = None
                 
    def fit(self, X, y = None):
                 self.scaler.fit(X[self.columns], y)
                 self.mean_ = np.mean(X[self.columns])
                 self.var_ = np.var(X[self.columns])
                 return self
                 
    def transform(self, X, y = None, copy = None):
                 init_col_order = X.columns
                 X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns = self.columns)
                 X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
                 return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

# first function within the custom scaler will not standardise all inputs, but only the ones we choose
# the dummies will be preserved


# In[20]:


unscaled_inputs.columns.values


# In[21]:


#columns_to_scale = ['Month Value', 'Day of the Week', 'Transportation Expense', 'Distance to Work',
       #'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children', 'Pets']

columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']

columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]

# columns to omit is a list that contains the 4 dummy variables (reasons) and education 
#'columns to scale' is a list that contains the names of the features we'd like to scale
# the list contains an iterative function that takes any values from unscaled_inputs if they are not already in columns to omit
# after we have manually placed the dummy variables (reason 1,2,3,4) into the 'columns_to_omit'


# In[22]:


absenteeism_scaler = CustomScaler(columns_to_scale)
# this variable just created will be used to subtract the mean and divide by the SD (feature-wise)


# In[23]:


absenteeism_scaler.fit(unscaled_inputs)
# from now, whenever we get new data, we know that the standardization info is stored in 'Absenteeism_scaler'


# In[24]:


# we have just prepared the scaling mechanism, now we need to transform to standardise the inputs


# In[25]:


scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


# In[26]:


scaled_inputs.shape


# ## Splitting the data for training and testing, and shuffling!

# In[27]:


from sklearn.model_selection import train_test_split
# this module splits arrays or matrices into random train and test subsets


# ### Split

# In[28]:


train_test_split(scaled_inputs, targets)


# In[29]:


# the resulting data is split into 4 arrays:
# training dataset with inputs
# training dataset with targets
# test dataset with inputs
# test dataset with targets

# we will set the training data size to 80% or 0.8
# also to shuffle we want to set the random_state to a number, so the values aren't shuffled every time we run the code
# if we didn't set it and it was reshuffled every time, the results would differ each time the code is run
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, train_size = 0.8, random_state = 20)


# In[30]:


print (x_train.shape, y_train.shape)
# this tells us that the inputs contain 560 observations along 12 features (variables)
# and the targets are a vector of length 560


# In[31]:


print (x_test.shape, y_test.shape)
# same for the test inputs
# this shows that we have 80% of observations for training, and 20% for testing (20% of 700 is 140, 80% is 560)


# ## Logistic Regression with sklearn

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# ### Training the model

# In[33]:


reg = LogisticRegression()


# In[34]:


reg.fit(x_train, y_train)
reg.get_params()


# In[35]:


reg.score(x_train, y_train)


# In[36]:


# based on the data we used, our model was able to classify around 78% of the observations correctly
# therefore, it is 78% accurate


# ### Manually Check the accuracy

# In[37]:


# good to have a full understanding of what we are doing here
model_outputs = reg.predict(x_train)
model_outputs


# In[38]:


y_train


# In[39]:


model_outputs == y_train
# shows how many of our model's outputs match the target output (True if match, False if not)


# In[40]:


np.sum((model_outputs==y_train))
# this gives us the total number of true entries (correct prediction)


# In[41]:


# this gives us the the total number of entries
model_outputs.shape[0]


# In[42]:


# to get the accuracy, we divide the number of correct predictions by the number of entries
np.sum((model_outputs==y_train)) / model_outputs.shape[0]


# ## Find the Intercept and the Coefficient

# In[43]:


reg.intercept_


# In[44]:


reg.coef_


# In[45]:


# but we want to know what variable these coefficients refer to
unscaled_inputs.columns.values


# In[46]:


# we should declare a new variable that contains this info
feature_name = unscaled_inputs.columns.values


# In[47]:


# now we want to create a uniqe dataframe that includes the intercept, the feature names and the corresponding coefficients
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)
# here the dataframe has one column (Feature name) which data comes from the variable (feature_name)

summary_table['Coefficient'] = np.transpose(reg.coef_)
# here we match those names with the coefficients, note: we must transpose this array because by default,
# ... ndarrays are rows and not columns

summary_table


# In[48]:


# now we need to add the intercept

summary_table.index = summary_table.index + 1
# this shifts up all incices by 1 so the 0th is empty
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# ### Interpreting the Coefficients

# In[49]:


# Standardised coefficients are basically the coefficients of a regression where all the variables have been standardised

summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)
# odds ratio is the term for what we will get after standardising the coefficients


# In[50]:


summary_table.sort_values('Odds_ratio', ascending=False)
# this sorts the table by the value of the odds ratio


# ### Summary Table Explained

# In[51]:


##### a feature is NOT important if it's coefficient is close to 0 nor if it's odds ratio is close to 1
##### the further away from 0 a coefficient is, the biggger that feature's importance is
##### a weight (coefficient) of 0 implies that no matter the feature, we multiply it by 0 in the model
##### for a unit change in the standardized feature, the odds increase by a multiple equal to the odds ratio (1 = no change)

##### if odds were 5:1 and odds ratio was 2 then new odds for a single unit change will be 10:1
##### if odds were 5:1 and odds ratio was 0.2 then new odds for a single unit change will be 1:1
##### if odds were 5:1 and odds ratio was 1 then new odds for a single unit change will be 5:1 as odds haven't changed

##### e.g. above we can see that Feature name 10 = Daily Work Load Average has coef of -0.004 and odds_ratio of almost 1
##### therefore this feature is almost useful for our model, our result will likely be the same without it
##### therefore it was removed during backward elimination (that's why it's not there anymore)

##### similarly with feature 8 and 6 (distance to work and day of the week) also make very little difference


# In[52]:


# according to the table, the most crucial reason for obsessive absence is Reason 3( poisoned),
# the odds of this being the reason for absence are 22 times higher than when no reason was given (reason 0)

# the second most crucial is Reason 1 (various diseases), the odds of this being the reason is around 16 times more likely
# than when no reason was given (reason 0)

# the third is reason 2 (pregnancy and giving birth) which is around 2.5 times more likely than
# when no reason was given. Significantly smaller factor than the first 2 reasons

# Transportation expense is the most importout out of the non-dummy features of the model
# however, it is one of our standardised variables so we can't interpret it directly here

# lets look at one negative coefficient e.g. Pets variable odd's ratio is 0.75
# so for each standardised unit of pet, the odds ratio are 25% lower than the base model(no pet)

# the intercept (bias) is used to get more accurate predictions, we can't actually interpret it but
# it calibrates the model (helps to standardise it)
# without the intercept, each prediction will be off the mark by precisely that value


# #### Backward Elimination

# In[53]:


# the idea is to simplify our model by removing all features which have close to no contribution to the model.
# e.g. when we have p-values, we get rid of all coefficients with p-values above > 0.05
# when using sklearn we don't have p-values as they aren't really needed
# if the weight is small enough, it won't make a difference
# if we remove these variables, the rest of our model won't change in terms of it's coefficient values


# ## Testing the Model

# In[54]:


reg.score(x_test, y_test)
# based on data the model has never seen before, 75% of the cases the model will predict correctly if a person is going
# to be excessively absent
# usually the test accuracy is 10-20% lower than the train accuracy (due to overfitting)


# In[55]:


# now lets show the probability estimates for all possible outputs (classes)
predicted_proba = reg.predict_proba(x_test)
predicted_proba
# the array shows the probability our model assigned to the observation being 0 (on the left)
# and the probability our model assigned to the observation being 1 (on the right)
# here we are only interested the assigned probability of excessive absenteeism (on the right)(1)


# In[58]:


predicted_proba[:,1]


# ### Save the Model

# In[59]:


import pickle
# pickle is a module used to convert a Python object into a string of characters that can be saved in a file
# we are using it here to save our model for use in other analysis
# we will unpickle it later in different software to convert it back into a Python object


# In[60]:


with open('model', 'wb') as file:
    pickle.dump(reg, file)
# model is the file name, and wb stands for 'write bytes' when we unpickle we will use rb (read bytes)
# the 'dump' method saves the following arguments () into a file
# here we are now separating the model from the training data so it is no longer reliant on it


# In[62]:


# here we are also saving the absenteeism_scaler to preprocess new data
with open('scaler', 'wb') as file:
    pickle.dump(absenteeism_scaler, file)


# In[ ]:


# another step of the deployment is creating a mechanism to load the saved model and make predictions

