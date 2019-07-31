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



rf = RandomForestClassifier(n_estimators=50, max_depth=50, n_jobs= -1, random_state=0, class_weight= 'balanced')
#Load Data
X_data = pd.read_csv('data/cleaned_data.csv', index_col= 0)
y_data = pd.read_csv('data/target_data.csv', index_col= 0)

#Train test and split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data['treatment'], test_size=0.33, random_state=42)

#Fit the model
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
