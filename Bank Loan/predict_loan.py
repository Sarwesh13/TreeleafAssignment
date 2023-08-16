import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


data = pd.read_excel('New Bank_loan_data.xlsx')

#handling empty and anamolous values

data['Home Ownership'].fillna('None', inplace=True) #home ownership has some blanks

data['Gender'].replace('-', 'Undisclosed', inplace=True) #gender has '-'
data['Gender'].replace('#', 'Undisclosed', inplace=True) #gender also has '#'
data['Gender'].fillna('Undisclosed', inplace=True) #gender also has blanks

data['Experience'].replace(-1, 0, inplace=True)
data['Experience'].replace(-2, 0, inplace=True) #experience has -1,-2,and-3 values
data['Experience'].replace(-3, 0, inplace=True)

data['Income'].fillna(0, inplace=True) #income has some blanks

data['Online'].fillna(0, inplace=True)#online has some blanks


data = data[data['Personal Loan'] != ' '] #remove any empty rows on loan.
data = data[(data['Age'] >= 1) & (data['Age'] <= 100)] #some ages are 3-digit, which are not possible
data['Personal Loan'] = data['Personal Loan'].astype(int) #personal loan's values weren't all integer

#label encoding datas because 'Gender' and 'Home Ownership' are not numerical values
label_encoder_gender = LabelEncoder()
label_encoder_home_ownership = LabelEncoder()
label_encoder_gender.fit(data['Gender'])
label_encoder_home_ownership.fit(data['Home Ownership'])
data['Gender'] = label_encoder_gender.transform(data['Gender'])
data['Home Ownership'] = label_encoder_home_ownership.transform(data['Home Ownership'])

#x-features, y-target
X = data.drop(['ID', 'Personal Loan'], axis=1)
y = data['Personal Loan']

#training and testing data splits (80% and 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#standardize 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#training random forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

#model evaluation
train_accuracy = rf_classifier.score(X_train_scaled, y_train)
test_accuracy = rf_classifier.score(X_test_scaled, y_test)
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")

#save and load trained model
joblib.dump(rf_classifier, 'loan_classifier_model.pkl')
loaded_model = joblib.load('loan_classifier_model.pkl')

#prediction(e.g. with high probability of Accepting Loan (1))
predict1 = pd.DataFrame({
    'Age': [30],
    'Gender': ['F'],
    'Experience': [28],
    'Income': [60],
    'ZIP Code': [92121],
    'Family': [3],
    'CCAvg': [8.9],
    'Education': [3],
    'Mortgage': [0],
    'Home Ownership': ['Home Owner'],
    'Securities Account': [0],
    'CD Account': [0],
    'Online': [0],  
    'CreditCard': [0]
})
#prediction(e.g. with high probability of Rejecting Loan (0))
predict2 = pd.DataFrame({
    'Age': [28],
    'Gender': ['Undisclosed'],
    'Experience': [8],
    'Income': [60],
    'ZIP Code': [92121],  
    'Family': [3],
    'CCAvg': [2.5],
    'Education': [1],
    'Mortgage': [100],
    'Home Ownership': ['None'],
    'Securities Account': [0],
    'CD Account': [0],
    'Online': [0],  
    'CreditCard': [1]
})

#preprocess the example data
predict1['Gender'] = label_encoder_gender.transform(predict1['Gender'])
predict1['Home Ownership'] = label_encoder_home_ownership.transform(predict1['Home Ownership'])
predict1_scaled = scaler.transform(predict1)

predict2['Gender'] = label_encoder_gender.transform(predict2['Gender'])
predict2['Home Ownership'] = label_encoder_home_ownership.transform(predict2['Home Ownership'])
predict2_scaled = scaler.transform(predict2)

#make predictions for above 2 datas
prediction1 = loaded_model.predict(predict1_scaled)
prediction2 = loaded_model.predict(predict2_scaled)

if prediction1 == 1:
    print(f"prediction1: {prediction1} Accepted!")
else:
    print(f"prediction1: {prediction1} Rejected.")

if prediction2 == 1:
    print(f"prediction2: {prediction2} Accepted!")
else:
    print(f"prediction2: {prediction2} Rejected.")

#EDA: Distribution plots for numeric variables
numeric_cols = ['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage']
sns.set(style="whitegrid")
for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()