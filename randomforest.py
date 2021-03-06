from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn import metrics

def vary_trees_graph(tree_values, X_train, X_test, y_train, y_test):
    results = []
    for tree in tree_values:
        RFC = RandomForestClassifier(n_estimators = tree, n_jobs = -1)
        model = RFC.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results.append(accuracy)
#     return results
    plt.figure(figsize= (8, 6))
    plt.title("Impact of Trees on Accuracy")
    sns.lineplot(x=tree_values, y=results)
    plt.xlabel("Number of Trees", fontsize= 16)
    plt.ylabel("Accuracy", fontsize= 16)
    plt.show();

def vary_depth(depth_values, X_train, X_test, y_train, y_test):
    results = []
    for tree in depth_values:
        RFC = RandomForestClassifier(n_estimators = 50, max_depth = tree, n_jobs = -1, warm_start = True)
        model = RFC.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results.append(accuracy)
#     return results
    plt.figure(figsize= (8, 6))
    plt.title("Impact of Trees on Accuracy")
    sns.lineplot(x=depth_values, y=results)
    plt.xlabel("Depth of Trees", fontsize= 16)
    plt.ylabel("Accuracy", fontsize= 16)
    plt.show();


if __name__ == "__main__":
    rf = RandomForestClassifier(n_estimators=50, max_depth=50, n_jobs= -1, random_state=0, class_weight= 'balanced')
#Load Data
    X_data = pd.read_csv('data/cleaned_data.csv', index_col= 0)
    y_data = pd.read_csv('data/target_data.csv', index_col= 0)

#Train test and split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data['treatment'], test_size=0.33, random_state=42)

#Train hyper parameters
    trees = [x for x in range(50, 1000, 50)]
    deep = [x for x in range(10, 500, 10)]
    vary_trees_graph(trees, X_train, X_test, y_train, y_test)
    vary_depth(deep, X_train, X_test, y_train, y_test)

#Fit the model
    rf = RandomForestClassifier(n_estimators=100, max_depth=100, n_jobs= -1, random_state=0, class_weight= 'balanced')

    rf = rf.fit(X_train, y_train)

#Open pickle and save model
    best_rf_model = open('models/best_rf_model.pkl', 'wb')
    pickle.dump(rf, best_rf_model)

    #Pull out features
    features = rf.feature_importances_
    feature_names = X_data.columns
    df_importances = pd.DataFrame(feature_names)
    df_importances['features'] = features
    df_importances.sort_values('features',ascending=False,inplace=True)

    #Feature importance barplot
    plt.figure(figsize= (12,8))
    sns.barplot(df_importances['features'][:10], feature_names[:10])
    plt.tight_layout()

    #Metrics
    y_pred=rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_test, y_pred=y_pred)
    recall = metrics.recall_score(y_true=y_test, y_pred=y_pred)


    print('accuracy: {:.3f}'.format(accuracy))
    print('precision: {:.3f}'.format(precision))
    print('recall: {:.3f}'.format(recall))
