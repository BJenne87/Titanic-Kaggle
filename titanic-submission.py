import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv("/home/PapaIV/Desktop/Python/Titanic/train.csv")
test_data = pd.read_csv("/home/PapaIV/Desktop/Python/Titanic/test.csv")

"""
preprocess categorical data to integers.
 sex: male= 0, female= 1
 port: Cherbourg= 0, Queenstown= 1, Southampton= 2
"""
int_categories = {'male': 0, 'female': 1, 'C': 0, 'Q': 1, 'S': 2}
train_data = train_data.replace(int_categories)
test_data = test_data.replace(int_categories)

# replace np.nan to pd.na so embarked can be an int
train_data['Embarked'] = train_data['Embarked'].astype('Int64')
test_data['Embarked'] = test_data['Embarked'].astype('Int64')

print(train_data)

# feature engineer
"""
some ideas to look into for features
do the passengers have complete information - bool. Maybe survivors have more records
"""
complete_train = train_data.apply(lambda row: 1 if pd.isna(row).any() == False else 0, axis=1)
complete_test = test_data.apply(lambda row: 1 if pd.isna(row).any() == False else 0, axis=1)
train_data['complete'] = complete_train
test_data['complete'] = complete_test

# model - Logistic regression for categorical results
features = ['complete', 'Pclass', 'Sex']
x = train_data[features]
y = train_data['Survived']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,) # split the data to test

#initialize and fir the model
model = LogisticRegression()
model.fit(x_train,y_train)

# predict outcomes
y_pred = model.predict(x_test)

# Print a confusion matrix to analyze type I and II errors
print(confusion_matrix(y_test, y_pred))

#ChatGPT recommended this report, look into it further
print(classification_report(y_test, y_pred)) 

# run the model on the test data and submit
x_predict = test_data[features]
y_predict = model.predict(x_predict)

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': y_predict
})
submission.to_csv('submission.csv', index=False)
