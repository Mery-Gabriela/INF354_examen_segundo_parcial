from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

import csv

dataset = []
prediccion_correcta = []

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
                prediccion_correcta.append(map_values[i].index(row[i]))
            else:
                x.append(map_values[i].index(row[i]))
        dataset.append(x)

train_len = int(len(dataset) * 0.8)

x_train = dataset[:train_len]
x_test = dataset[train_len:]
y_true_train = prediccion_correcta[:train_len]
y_true_test = prediccion_correcta[train_len:]

kmeans = KMeans(n_clusters = 2)
kmeans.fit(x_train)

y_train = kmeans.labels_

print(y_true_train)
print(y_train)

conf_matrix = confusion_matrix(y_true_train, y_train)

print(conf_matrix)

############################### https://en.wikipedia.org/wiki/Confusion_matrix

y_test = kmeans.predict(x_test)
conf_matrix2 = confusion_matrix(y_true_test, y_test)

print(conf_matrix2)