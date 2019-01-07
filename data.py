import pandas as pd
import numpy as np
from keras import models
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from keras import layers



df = pd.read_csv('data/data_adult_clean_1.csv')
x_train = df.iloc[:,:7].join(df.iloc[:,8:])

y_train = df.iloc[:,7]


x_test = x_train.iloc[20000:,:]
y_test = y_train[20000:]


x_train = x_train.iloc[:20000,:].astype('float32')

y_train = y_train[:20000]



y_train = np.asarray(y_train).astype('float32')
y_test = np.asarray(y_test).astype('float32')

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


model = models.Sequential()
model.add(layers.Dense(50,activation='sigmoid',input_shape=(x_train.shape[1],)))
#model.add(layers.Dense(30,activation='sigmoid'))
model.add(layers.Dense(30,activation='sigmoid'))
model.add(layers.Dense(10,activation='sigmoid'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.rmsprop(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=100  ,batch_size=512)

results = model.evaluate(x_test,y_test)

results

# output : [0.7323556204136373, 0.8061405234720688] 80.61% accuracy on test data
