import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def main():
    train_data = pd.read_csv('train.csv') # construction of the dataframe
    test_data = pd.read_csv('test.csv')

    train_data.head()


    train_data = train_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1) #dropping useless columns
    test_data = test_data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1) #axis 1 specifies columns

    total = [train_data, test_data]
    for dataset in total: # creating age range classifications so that it better fits the 0,1
        dataset.loc[dataset['Age'] <= 18, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[dataset['Age'] > 64, 'Age'] = 4

    total = [train_data, test_data]
    train_data.Embarked.value_counts()
    eb_rmv_pak = train_data.Embarked.mode() #replacing empty embarked with most occuring
    ag_rmv_pak = train_data.Age.mode() #same for age
    for dataset in total:
        dataset['Age'] = dataset.Age.fillna(ag_rmv_pak[0]) #removed na age values replaced with highest occuring
        dataset['Embarked'] = dataset.Embarked.fillna(eb_rmv_pak[0])

    train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True) # sex changed to 0 and 1
    test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

    print(train_data.corr()) #prints correlation table

    av_Fare = test_data.Fare.mean() #replaces empty values in fare with the mean fare value
    test_data['Fare'] = test_data.Fare.fillna(av_Fare)




    df_1 = train_data.loc[:, ['Fare', 'Pclass']] #creating dataframe of fare and p class
    df_2 = test_data.loc[:, ['Fare', 'Pclass']]
    pca = PCA(n_components=1)
    ins1 = pca.fit_transform(df_1)
    ins2 = pca.fit_transform(df_2)

    train_data['proj1'] = ins1[:, 0] #reinsert new projection vectors back into data set
    test_data['proj1'] = ins2[:, 0]

    train_data = train_data.drop(['Fare', 'Pclass'], axis=1) # dropping old columns after reinsertation
    test_data = test_data.drop(['Fare', 'Pclass'], axis=1)

    df_3 = train_data.loc[:, ['SibSp', 'Parch']]#same
    df_4 = test_data.loc[:, ['SibSp', 'Parch']]
    pca = PCA(n_components=1)
    ins3 = pca.fit_transform(df_3)
    ins4 = pca.fit_transform(df_4)

    train_data['proj2'] = ins3[:, 0]
    test_data['proj2'] = ins4[:, 0]

    train_data = train_data.drop(['SibSp', 'Parch'], axis=1)
    test_data = test_data.drop(['SibSp', 'Parch'], axis=1)

    train_data.info() # checking dataframe after replacements

    x_train = train_data.drop('Survived', axis=1) #x is the input so you dont want predictor values
    y_train = train_data['Survived']


    np.where(x_train >= np.finfo(np.float64).max) #dont store things with too high percision



    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    Y_pred_1 = logreg.predict(test_data)
    acc_log = logreg.score(x_train, y_train) * 100
    print(acc_log)  # percent of test data it got correct.


    svc = SVC()
    svc.fit(x_train,y_train)
    y_pred_2 = svc.predict(test_data)
    acc_svc = svc.score(x_train,y_train)*100
    print(acc_svc)


    decision_tree = DecisionTreeClassifier() #DTs overfit. good when data isnt biased torwards specific cat
    decision_tree.fit (x_train, y_train)
    y_pred_3 = decision_tree.predict(test_data)
    acc_tree= decision_tree.score(x_train,y_train)*100
    print(acc_tree)

    random_forest = RandomForestClassifier() #highest accuracy while not overfitting
    random_forest.fit(x_train,y_train)
    y_pred_4 = random_forest.predict(test_data)
    acc_forest = random_forest.score(x_train,y_train)*100
    print(acc_forest)
    scores = []
    indexes = []




    for i in range(1,100):
        K_neighbor = KNeighborsClassifier(n_neighbors=i)
        K_neighbor.fit(x_train, y_train)
        y_pred_5 = K_neighbor.predict(test_data)
        acc_neighbor = K_neighbor.score(x_train, y_train) * 100
        scores.append(acc_neighbor)
        indexes.append(i)

    plt.scatter(indexes,scores)
    plt.show()




if __name__== "__main__":
    main()




