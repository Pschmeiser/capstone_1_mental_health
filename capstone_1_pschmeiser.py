# %%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from scipy.stats import ttest_ind
from numpy import sqrt
%matplotlib inline
plt.style.use('ggplot')
# %%
mental_df = pd.read_csv('mental_health.csv')
mental_df.head()
# %%
mental_df.info()
# %%
mental_df = mental_df.rename(columns={'Have you sought treatment for a mental health condition?': 'treatment', 'How many employees does your company or organization have?': 'size', 'Do you have a family history of mental illness?': 'history'})
# %%
mental_df['Gender'] = mental_df['Gender'].replace('Male','M')
mental_df['Gender'] = mental_df['Gender'].replace('male','M')
mental_df['Gender'] = mental_df['Gender'].replace('m','M')
mental_df['Gender'] = mental_df['Gender'].replace('female','F')
mental_df['Gender'] = mental_df['Gender'].replace('Female','F')
mental_df['Gender'] = mental_df['Gender'].replace('f','F')
mental_df['Gender'] = mental_df['Gender'].replace('ostensibly male, unsure what that really means','M')
mental_df['Gender'] = mental_df['Gender'].replace('Cis Man','M')
mental_df['Gender'] = mental_df['Gender'].replace('Male-ish','M')
mental_df['Gender'] = mental_df['Gender'].replace('maile','M')
mental_df['Gender'] = mental_df['Gender'].replace('something kinda male','M')
mental_df['Gender'] = mental_df['Gender'].replace('Cis Male','M')
mental_df['Gender'] = mental_df['Gender'].replace('Mal','M')
mental_df['Gender'] = mental_df['Gender'].replace('Male (CIS)','M')
mental_df['Gender'] = mental_df['Gender'].replace('something kinda male?','M')
mental_df['Gender'] = mental_df['Gender'].replace('Make','M')
mental_df['Gender'] = mental_df['Gender'].replace('Guy (-ish) ^_^','M')
mental_df['Gender'] = mental_df['Gender'].replace('Male ','M')
mental_df['Gender'] = mental_df['Gender'].replace('Man','M')
mental_df['Gender'] = mental_df['Gender'].replace('msle','M')
mental_df['Gender'] = mental_df['Gender'].replace('Mail','M')
mental_df['Gender'] = mental_df['Gender'].replace('cis male','M')
mental_df['Gender'] = mental_df['Gender'].replace('Malr','M')
mental_df['Gender'] = mental_df['Gender'].replace('Cis Female','F')
mental_df['Gender'] = mental_df['Gender'].replace('Woman','F')
mental_df['Gender'] = mental_df['Gender'].replace('Female (cis)','F')
mental_df['Gender'] = mental_df['Gender'].replace('femail','F')
mental_df['Gender'] = mental_df['Gender'].replace('Female','F')
mental_df['Gender'] = mental_df['Gender'].replace('Femake','F')
mental_df['Gender'] = mental_df['Gender'].replace('woman','F')
mental_df['Gender'] = mental_df['Gender'].replace('cis-female/femme','F')

# %%
mask_M = mental_df["Gender"] == "M"
mask_F = mental_df["Gender"] == "F"
mask_mf = mask_M | mask_F
mental_gender = mental_df[mask_mf]
# %%
mental_gender = mental_gender[["Gender", "treatment"]]
mental_gender.head()
# %%
mental_size = mental_df[['size','treatment']]
mental_size.head()
# %%
mental_history = mental_df[['history','treatment']]
mental_history.head()
# %%
mental_size['Mental_illness_yes'] = pd.Series(np.where(mental_size["treatment"] == "Yes",1,0), mental_size.index)
size_avg = mental_size.groupby('size').mean()
size_avg
# %%
order = [1,4,3,5,2,6]
size_avg['order']=order
size_avg = size_avg.sort_values(['order'])
size_avg
# %%
sns.set(font_scale=2.0)
# %%
plt.figure(figsize=(15,10))
sns.barplot(x=size_avg.index.values, y=size_avg['Mental_illness_yes'])
plt.xticks(rotation= 45)
plt.xlabel('Company Size')
plt.ylabel('Percent Mental Illness')
plt.title('Occurance of Mental Illness by Company Size')
# %%
mental_history['Mental_illness_yes'] = pd.Series(np.where(mental_history["treatment"] == "Yes",1,0), mental_history.index)
history_avg = mental_history.groupby('history').mean()
history_avg
# %%
plt.figure(figsize=(15,10))
sns.barplot(x=history_avg.index.values, y=history_avg['Mental_illness_yes'])
plt.xticks(rotation= 45)
plt.xlabel('History of Mental Illness')
plt.ylabel('Percent Mental Illness')
plt.title('Occurance of Mental Illness Based on Family History of Mental Illness')
# %%
history_avg_t = mental_history.mean()
history_avg_t = history_avg_t.values.tolist()
history_avg_t = history_avg_t[0]
history_avg_t
# %%
no_avg = history_avg.values.tolist()
no_avg = no_avg[0][0]
no_avg
# %%
yes_avg = history_avg.values.tolist()
yes_avg = yes_avg[1][0]
yes_avg
# %%
no_sum = mental_history[mental_history['history']=='No'].count()
no_sum = no_sum['history']
no_sum
# %%
yes_sum = mental_history[mental_history['history']=='Yes'].count()
yes_sum = yes_sum['history']
yes_sum
# %%
se_his = sqrt(history_avg_t * (1 - history_avg_t) * (1 / no_sum + 1 / yes_sum))
z_his =(yes_avg-no_avg)/se_his
z_his
# %%
p_val_his = 1 - stats.norm.cdf(z_his)
p_val_his
# %%
alpha = 0.05
reject_null = p_val_his < alpha
reject_null
# %%
mental_gender['Mental_illness_yes'] = pd.Series(np.where(mental_gender["treatment"] == "Yes",1,0), mental_gender.index)
gender_avg = mental_gender.groupby('Gender').mean()
gender_avg
# %%
plt.figure(figsize=(15,10))
sns.barplot(x=gender_avg.index.values, y=gender_avg['Mental_illness_yes'])
plt.xticks(rotation= 45)
plt.xlabel('Sex')
plt.ylabel('Percent Mental Illness')
plt.title('Occurance of Mental Illness by Sex')
# %%
gender_avg_t = mental_gender.mean()
gender_avg_t = gender_avg_t.values.tolist()
gender_avg_t = gender_avg_t[0]
gender_avg_t
# %%
fem_avg = gender_avg.values.tolist()
fem_avg = fem_avg[0][0]
fem_avg
# %%
male_avg = gender_avg.values.tolist()
male_avg = male_avg[1][0]
male_avg
# %%
fem_sum = mental_gender[mental_gender['Gender']=='F'].count()
fem_sum = fem_sum['Gender']
fem_sum
# %%
male_sum = mental_gender[mental_gender['Gender']=='M'].count()
male_sum = male_sum['Gender']
male_sum
# %%
se_gen = sqrt(gender_avg_t * (1 - gender_avg_t) * (1 / fem_sum + 1 / male_sum))
z_gen =(fem_avg-male_avg)/se_gen
z_gen
# %%
p_val_gen = 1 - stats.norm.cdf(z_gen)
p_val_gen
# %%
alpha = 0.05
reject_null = p_val_gen < alpha
reject_null
# %%
