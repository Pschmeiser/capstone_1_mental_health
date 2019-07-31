from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
plt.rcParams.update({'font.size': 14})
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import metrics

from sklearn.inspection.partial_dependence import partial_dependence, plot_partial_dependence

def calculate_stats(model, X_train, X_test, y_train, y_test):
    trained = model.fit(X_train, y_train)
    accuracy = trained.score(X_test, y_test)
    y_pred = trained.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict= True)
    probs = trained.predict_proba(X_test)
    #precision = report['True']['precision']
    #recall = report['True']['recall']
    print(report)
    return probs

def vary_trees_graph(tree_values, X_train, X_test, y_train, y_test):
    results = []
    for tree in tree_values:
        clf = GradientBoostingClassifier(learning_rate = tree_values)
        model = clf.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results.append(accuracy)
#     return results
    plt.figure(figsize= (8, 6))
    plt.title("Impact of Trees on Accuracy")
    sns.lineplot(x=tree_values, y=results)
    plt.xlabel("Number of Trees", fontsize= 16)
    plt.ylabel("Accuracy", fontsize= 16)
    plt.show();

def vary_trees_depth(depth_values, X_train, X_test, y_train, y_test):
    results = []
    for tree in depth_values:
        clf = GradientBoostingClassifier(n_estimators = 50, max_depth = tree, n_jobs = -1, warm_start = True)
        model = clf.fit(X_train, y_train)
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
    rf = GradientBoostingClassifier()
    X_data = pd.read_csv('data/cleaned_data.csv', index_col= 0)
    y_data = pd.read_csv('data/target_data.csv', index_col= 0)
    X_data.head()

    #X_data.drop('Unnamed: 0', axis= 1)

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data['treatment'], test_size=0.33, random_state=42)
    X_train.shape
    y_train.shape



    # tune hyper-parameters
    trees = [x for x in range(50, 1000, 50)]

    deep = [x for x in range(10, 500, 10)]

    # vary_trees_graph(trees, X_train_SMOTE, X_test_SMOTE, y_train_SMOTE, y_test_SMOTE)
    # vary_trees_depth(deep, X_train_SMOTE, X_test_SMOTE, y_train_SMOTE, y_test_SMOTE)

    vary_trees_graph(trees, X_train, X_test, y_train, y_test)
    vary_trees_depth(deep, X_train, X_test, y_train, y_test)



    # test model performance
    sampled_probs = calculate_stats(rf, X_train, X_test, y_train, y_test)


    best_rf_model = open('models/best_rf_model.pkl', 'wb')
    pickle.dump(rf, best_rf_model)
    # Close the pickle instances
    best_rf_model.close()
    features = rf.feature_importances_
    features.shape
    feature_names = X_data.columns
    plt.figure(figsize= (12,8))
    sns.barplot(features, feature_names)
    feature = zip(features,feature_names)
    feature = sorted(feature)
    plt.figure(figsize=(12,8))
    sns.boxplot(feature)

    my_plots = plot_partial_dependence(rf, features=[3], X=X_data, feature_names=X_data.columns , grid_resolution=10)
    plt.tight_layout()
    plt.savefig('images/pdp3.png')

    Table = pd.DataFrame()
    columns =   ['Precision','Recall','F1']
    Table = pd.DataFrame(columns, axis = 1)
    Table.head()
    y_pred=rf.predict(X_test)
    accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = metrics.precision_score(y_true=y_test, y_pred=y_pred)
    recall = metrics.recall_score(y_true=y_test, y_pred=y_pred)
    classification_report= metrics.classification_report(y_true=y_test, y_pred=y_pred)

    print('accuracy: {:.3f}'.format(accuracy))
    print('precision: {:.3f}'.format(precision))
    print('recall: {:.3f}'.format(recall))
    print('=========================================================')
    print('classification_report: \n{}'.format(classification_report))

    table = pd.DataFrame(columns=['Accuracy','Precision','Recall'])
    table.head()
    table.append(accuracy)
