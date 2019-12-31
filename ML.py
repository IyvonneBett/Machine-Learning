import sklearn
import pandas
import matplotlib.pyplot as plt

df = pandas.read_csv('banksdata.csv')
print(df)
#print(df.isnull().sum()) # to show empties gender 65 empty
# ML models dont work with empty records and text
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

# Loan_Status was not replaced since it was the target variable
# model learns from 0 to 10 last is target
# we split features and the labels
array = df.values # we read all data into an array

features = array[:, 0:11] # : colon -All rows ....11 is not counted here
target = array[:, 11] # 11th column which is the loan Status is counted here

# we will use only 70% of data to train our model 2100/3000 predict remaining 900
# we will use 30% to test the model....900

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, labels, test_size=0.3, random_state=42) # random set of 42 test size 30%


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
print('Model finished...')




