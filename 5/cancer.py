from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import csv

X = []
Y = []

map_values = [
    ['no-recurrence-events', 'recurrence-events'],
    ['10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'],
    ['lt40', 'ge40', 'premeno'],
    ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59'],
    ['0-2', '3-5', '6-8', '9-11', '12-14', '15-17', '18-20', '21-23', '24-26', '27-29', '30-32', '33-35', '36-39'],
    ['yes', 'no', '?'],
    ['1', '2', '3'],
    ['left', 'right'],
    ['left_up', 'left_low', 'right_up', 'right_low', 'central', '?'],
    ['yes', 'no']
]

with open('breast-cancer.data', newline='\n') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        x = []
        for i in range(len(row)):
            if i == 0:
                Y.append(map_values[i].index(row[i]))
            else:
                x.append(map_values[i].index(row[i]))
        X.append(x)

train_len = int(len(X) * 0.8)

x_train = X[:train_len]
x_test = X[train_len:]
y_train = Y[:train_len]
y_test = Y[train_len:]

kneighbors = KNeighborsClassifier(n_neighbors = 2)
kneighbors.fit(x_train, y_train)

out_test = kneighbors.predict(x_test)
# out_prob = kneighbors.predict_proba(x_test)

print(out_test)
# print(out_prob)
print(y_test)

conf_matrix = confusion_matrix(y_test, out_test)
print(conf_matrix)