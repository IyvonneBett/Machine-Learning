import random

import pandas
# Classification - supervised learning

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

df = pandas.read_csv("bank_updated1.csv")
print(df)
print(df.shape)
print(df.dtypes)
#
# # build a model for marketing team to use to predict members
# # who might subscribe to the offer
# print(df.isnull().sum())
#
# # we will work with below colms, we create a subset of the data frame
subset = df[['age','housing','loan','job','education','marital','balance','default','campaign','previous','y']]
print(subset)
#
subset['education'].replace({'primary':0, 'secondary':1,'tertiary':2, 'unknown':3}, inplace=True)
subset['marital'].replace({'married':0, 'single':1, 'divorced':2, 'unknown':3}, inplace=True)
subset['default'].replace({'yes':0, 'no':1, 'unknown':2}, inplace=True)
subset['job'].replace({'admin.':0,
'blue-collar':1,
'housemaid':2,
'entrepreneur':3,
'management':4,
'retired':5,
'self-employed':6,
'student':7,
'services': 8,
'technician':9,
'unemployed':10,
'unknown':11}, inplace=True)


subset['housing'].replace({'yes':0, 'no':1, 'unknown':2}, inplace=True)
subset['loan'].replace({'yes':0, 'no':1, 'unknown':2}, inplace=True)
print(subset.dtypes)


# split data

array = subset.values # convert the subset to an array
features = array[:, 0:10] # means 0 - 5 Input variables 10 is not counted
target = array[:,10] # 6th is counted here : output variables 10 is what we are predicting

#
# # # we use 70% for training...30% for testing
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features,
target,
test_size= 0.30,
random_state=42)

#
# # # classification models
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print('Training....finished')


predictions = model.predict(X_test)
print(predictions)


from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))


from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))


from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, predictions))



new_person = [[34,0, 0,4,1,1,2000,1, 3,6]]
observation = model.predict(new_person)
print(observation)