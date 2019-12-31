import sklearn
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv('banksdata.csv')
print(df)

df['Gender'].fillna('N', inplace= True)
df['Gender'].replace({'Male':0, 'Female':1, 'N':2}, inplace= True) #replace male and female with numbers

df['Married'].fillna('N', inplace= True)
df['Married'].replace({'No':0, 'Yes':1, 'N':2}, inplace= True)

df['Education'].replace({'Not Graduate':0, 'Graduate':1}, inplace= True) # education no empty

df['Self_Employed'].fillna('N', inplace= True)
df['Self_Employed'].replace({'No':0, 'Yes':1, 'N':2}, inplace= True)

df['Credit_History'].fillna(2, inplace= True) # credit history is already encoded with 0 and 1

df['Property_Area'].replace({'Urban':0, 'Rural':1, 'Semiurban':2}, inplace= True)

df['Dependents'].fillna(4, inplace= True)

medianLoan = df['LoanAmount'].median()
df['LoanAmount'].fillna(medianLoan, inplace=True)

medianTerm = df['Loan_Amount_Term'].median()
df['Loan_Amount_Term'].fillna(medianTerm, inplace=True)


print(df.isnull().sum())
print(df.dtypes)

array = df.values # we read all data into an array

features = array[:, 0:11] # : colon -All rows ....11 is not counted here
labels = array[:, 11] # 11th column which is the loan Status is counted here

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.neighbors import KNeighborsClassifier # model...algorithm
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print('Model finished...')

# ask the model to predict x_test features ...hide the y_test (Loan Status)
predictions = model.predict(X_test)
print(predictions)

#compare predictions and y_test
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, predictions))

from sklearn.metrics import classification_report
print(classification_report(Y_test, predictions))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(Y_test, predictions))
# read diagonally

new_member = [[0,1,2,1,0,3500,2000,250,360,1,1]] # from banksdata encoded
observation = model.predict(new_member)
print(observation)

