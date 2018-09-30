
# coding: utf-8

# In[60]:


import pandas as pd
import numpy as np
import matplotlib as plt

#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


df = pd.read_csv("C:/Users/kaush/Desktop/train.csv") #Reading the dataset in a dataframe using Pandas
test=pd.read_csv("C:/Users/kaush/Desktop/test.csv") 
df.head()
df.describe()
df['Property_Area'].value_counts()
#histogram plot of applicant income
df['ApplicantIncome'].hist(bins=50)
#boxplot of applicant income. This confirms the presence of a lot of outliers/extreme values
df.boxplot(column='ApplicantIncome')
#We can see that there is no substantial different between the mean income of graduate and non-graduates.
#But there are a higher number of graduates with very high incomes, which are appearing to be the outliers
df.boxplot(column='ApplicantIncome', by = 'Education')
#Again, there are some extreme values. 
#Clearly, both ApplicantIncome and LoanAmount require some amount of data munging
df['LoanAmount'].hist(bins=50)
df.boxplot(column='LoanAmount')

temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
print ('Frequency Table for Credit History:') 
print (temp1)
print ('\nProbility of getting loan for each Credit History class:')
print (temp2)

#Lets plot the above Credit history info. in a bar chart
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind = 'bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
#combining loanstatus and credit history in a stacked chart
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
#let us check the number of nulls / NaNs in the dataset
df.apply(lambda x: sum(x.isnull()),axis=0)
#lets replace null values using mean
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
#imputing the missing values as “No” as there is a high probability of success
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No',inplace=True)

# let’s try a log transformation to nullify the outliers effect
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
#combining appincome and coappincome to get total income
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 
#impute the missing values using mode
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

#Since, sklearn requires all inputs to be numeric.
#we should convert all our categorical variables into numeric by encoding the categories

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
pe= LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

df.dtypes 

# My HYPOTHESIS
#The chances of getting a loan will be higher for:
  # 1.Applicants having a credit history (remember we observed this in exploration?)
  # 2.Applicants with higher applicant and co-applicant incomes
  # 3.Applicants with higher education level
  #4.Properties in urban areas with high growth perspectives

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome])

    
#Accuracy : 80.945% Cross-Validation Score : 80.946%
outcome_var = 'Loan_Status'
model1 = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model1, df,predictor_var,outcome_var)
#Accuracy : 80.945% Cross-Validation Score : 80.946%
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model1, df,predictor_var,outcome_var)

#Accuracy : 81.930% Cross-Validation Score : 76.656%
model2 = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model2, df,predictor_var,outcome_var)
#Accuracy : 92.345% Cross-Validation Score : 71.009%
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model2, df,predictor_var,outcome_var)

#Accuracy : 100.000% Cross-Validation Score : 78.179%
#This is the ultimate case of overfitting

model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df,predictor_var,outcome_var)

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)

#Let’s use the top 5 variables for creating a model
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,predictor_var,outcome_var)
    


