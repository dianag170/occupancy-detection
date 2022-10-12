from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from knn_classifier import classify


data=pd.read_csv('Occupancy.csv',header=0)
data=data.dropna()
#print(data.shape)

data = data.drop(columns=['date'])
#print(list(data.columns))

data['Occupancy'].value_counts()

#sns.countplot(x='Occupancy',data=data,palette='hls')
#plt.show()

count_no_sub = len(data[data['Occupancy']==0])
count_sub = len(data[data['Occupancy']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
#print("percentage of no occupancy is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
#print("percentage of occupancy", pct_of_sub*100)


X_all_data = data.loc[:, data.columns != 'Occupancy']
y_all_labels = data.loc[:, data.columns == 'Occupancy']

# X_all_data = X_all_data.iloc[:1200,:]
# y_all_labels = y_all_labels.iloc[:1200,:]


fold_index = 0
k = 3
accuracy_scores = []


kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X_all_data):
    
    print("FOLD: ", fold_index)

    X_train, X_test = X_all_data.iloc[train_index,:], X_all_data.iloc[test_index,:]
    y_train, y_test = y_all_labels.iloc[train_index,:], y_all_labels.iloc[test_index,:]


    smote = SMOTE(random_state=0)

    #features
    columns = X_train.columns

    # SMOTE sampling of training data & labels
    smote_data_X, smote_labels_y = smote.fit_resample(X_train, y_train)

    smote_data_X = pd.DataFrame(data=smote_data_X,columns=columns )
    smote_labels_y= pd.DataFrame(data=smote_labels_y,columns=['Occupancy'])

    X_os_train=smote_data_X[columns]
    y_os_train=smote_labels_y['Occupancy']

    score = classify(X_os_train, X_test, y_os_train, y_test, k)
    accuracy_scores.append(score)
    print("Accuracy score wiht K =",k,"for fold",fold_index,"is",score)

    fold_index += 1

### Average accuracy score
accuracy_scores_avg = sum(accuracy_scores) / len(accuracy_scores) 
print("Average accuracy score for K =",k,"is:", accuracy_scores_avg)
