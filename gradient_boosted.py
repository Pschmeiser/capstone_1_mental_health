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


GB =GradientBoostingClassifier()
X_data = pd.read_csv('data/cleaned_data.csv', index_col= 0)
y_data = pd.read_csv('data/target_data.csv', index_col= 0)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data['treatment'], test_size=0.33, random_state=42)
GB = GB.fit(X_train, y_train)
best_rf_model = open('models/best_GB_model2.pkl', 'wb')
pickle.dump(GB, best_GB_model)
features = GB.feature_importances_
feature_names = X_data.columns
df_importances = pd.DataFrame(feature_names)
df_importances['features'] = features
df_importances.sort_values('features',ascending=False,inplace=True)
#Feature importance barplot
df_importances.head()
plt.figure(figsize= (12,8))
sns.barplot(df_importances['features'][:10], feature_names[:10])
plt.tight_layout()
