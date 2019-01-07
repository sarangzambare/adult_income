import pandas as pd
import numpy as np


df = pd.read_csv('data/data_adult_nounknown.csv')

df = df.drop('Unnamed: 0',axis=1)


df = df.join(pd.get_dummies(df.workclass))

df = df.drop('workclass',axis=1)

df = df.join(pd.get_dummies(df.education))
df = df.drop('education',axis=1)

df = df.join(pd.get_dummies(df.occupation))

df = df.drop('occupation',axis=1)

df = df.join(pd.get_dummies(df['marital-status']))

df = df.drop('marital-status',axis=1)

df = df.join(pd.get_dummies(df['relationship']))

df = df.drop('relationship',axis=1)

for i in range(df.shape[0]):
    df.iloc[i,4] = 0 if (df.iloc[i,4]==' Male') else 1


df = df.join(pd.get_dummies(df['race']))

df = df.drop('race',axis=1)

df = df.join(pd.get_dummies(df['native-country']))

df = df.drop('native-country',axis=1)


for i in range(df.shape[0]):
    df.iloc[i,7] = 0 if df.iloc[i,7]==' <=50K' else 1

df.to_csv('data/data_adult_clean_1.csv',index=False)
