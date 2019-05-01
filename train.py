import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
label = LabelEncoder()

train_df = pd.read_csv("./inputs/train.csv")
test_df = pd.read_csv("./inputs/test.csv")

test_df.Fare.fillna(test_df.Fare.mean(), inplace=True)
data_df = train_df.append(test_df)
passenger_id = test_df['PassengerId']

train_df.drop(['PassengerId'], axis=1, inplace=True)
test_df.drop(['PassengerId'], axis=1, inplace=True)

train_df['Sex'] = train_df.Sex.apply(lambda x: 0 if x == "female" else 1)
test_df['Sex'] = test_df.Sex.apply(lambda x: 0 if x == "female" else 1)

test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)

for name_string in data_df['Name']:
    data_df['Title'] = data_df['Name'].str.extract('([A-Za-z]+)\.', expand=True)

mapping = {
    'Mlle': 'Miss',
    'Major': 'Mr',
    'Col': 'Mr',
    'Sir': 'Mr',
    'Don': 'Mr',
    'Mme': 'Miss',
    'Jonkheer': 'Mr',
    'Lady': 'Mrs',
    'Capt': 'Mr',
    'Countess': 'Mrs',
    'Ms': 'Miss',
    'Dona': 'Mrs'}

data_df.replace({'Title': mapping}, inplace=True)

train_df['Title'] = data_df['Title'][:891]
test_df['Title'] = data_df['Title'][891:]

titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Rev', 'Dr']
for title in titles:
    age_to_impute = data_df.groupby('Title')['Age'].median()[titles.index(title)]
    data_df.loc[(data_df['Age'].isnull()) & (data_df['Title'] == title), 'Age'] = age_to_impute

train_df['Age'] = data_df['Age'][:891]
test_df['Age'] = data_df['Age'][891:]

test_df.drop(['Title', 'Cabin'], axis=1, inplace=True)
train_df.drop(['Title', 'Cabin'], axis=1, inplace=True)

train_df['Embarked'].fillna('S', inplace=True)
embarked_values = {'S': 1, 'C': 2, 'Q': 3}
train_df.replace({'Embarked': embarked_values}, inplace=True)
test_df.replace({'Embarked': embarked_values}, inplace=True)

X = train_df.drop(['Survived', 'Name', 'Ticket'], 1)
y = train_df['Survived']
X_test = test_df.copy()
X_test = X_test.drop(['Name', 'Ticket'], 1)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, y)
Y_pred = random_forest.predict(X_test)
random_forest.score(X, y)
acc_random_forest = round(random_forest.score(X, y) * 100, 2)
print(acc_random_forest)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X, y)
# Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X, y) * 100, 2)
print(acc_decision_tree)

logreg = LogisticRegression()
logreg.fit(X, y)
# Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X, y) * 100, 2)
print(acc_log)

gaussian = GaussianNB()
gaussian.fit(X, y)
# Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X, y) * 100, 2)
print(acc_gaussian)

perceptron = Perceptron()
perceptron.fit(X, y)
# Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X, y) * 100, 2)
print(acc_perceptron)

sgd = SGDClassifier()
sgd.fit(X, y)
# Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X, y) * 100, 2)
print(acc_sgd)

submission = pd.DataFrame({
    "PassengerId": passenger_id,
    "Survived": Y_pred
})
submission.to_csv('./submission_Random_Forest.csv', index=False)
