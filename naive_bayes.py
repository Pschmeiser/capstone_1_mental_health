import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_stats(model, X_train, X_test, y_train, y_test):
    trained = model.fit(X_train, y_train)
    accuracy = trained.score(X_test, y_test)
    y_pred = trained.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict= True)
    probs = trained.predict_proba(X_test)
#     precision = report['True']['precision']
#     recall = report['True']['recall']
    print(report)
    return probs
def vary_alpha_graph(hyper, X_train, X_test, y_train, y_test):
    results = []
    for value in hyper:
        clf = BernoulliNB(alpha=value, binarize=0, class_prior=None, fit_prior=True)
        model = clf.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results.append(accuracy)
#     return results
    plt.figure(figsize= (8, 6))
    plt.title("Impact of Alpha on Accuracy")
    sns.lineplot(x=hyper, y=results)
    plt.xlabel("Number of Alpha", fontsize= 16)
    plt.ylabel("Accuracy", fontsize= 16)
    plt.show();

def vary_bine_graph(hyper, X_train, X_test, y_train, y_test):
    results = []
    for value in hyper:
        clf = BernoulliNB(alpha=1.0, binarize=value, class_prior=None, fit_prior=True)
        model = clf.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        results.append(accuracy)
#     return results
    plt.figure(figsize= (8, 6))
    plt.title("Impact of Alpha on Accuracy")
    sns.lineplot(x=hyper, y=results)
    plt.xlabel("Number of Alpha", fontsize= 16)
    plt.ylabel("Accuracy", fontsize= 16)
    plt.show();

if __name__ == "__main__":
    Bern = BernoulliNB(alpha=1.0, binarize=0, class_prior=None, fit_prior=True)
    X_data = pd.read_csv('data/cleaned_data.csv')
    y_data = pd.read_csv('data/target_data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data['treatment'], test_size=0.33, random_state=42)

    sm = SMOTE(random_state=42)
    X_train_SMOTE, y_train_SMOTE = sm.fit_resample(X_train, y_train)
    X_test_SMOTE, y_test_SMOTE = sm.fit_resample(X_test, y_test)

    # tune hyper-parameters
    alpha = [1, .95, .9, .85, .8, .75, .7, .65, .6, .55, .5]
    bine = [0, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5]

    vary_alpha_graph(alpha, X_train_SMOTE, X_test_SMOTE, y_train_SMOTE, y_test_SMOTE)
    vary_bine_graph(bine, X_train_SMOTE, X_test_SMOTE, y_train_SMOTE, y_test_SMOTE)

    rf_SMOTE = Bern.fit(X_train_SMOTE, y_train_SMOTE)
    best_rf_model = open('models/best_rf_model.pkl', 'wb')
    pickle.dump(rf_SMOTE, best_rf_model)
    # Close the pickle instances
    best_rf_model.close()
