# -*- coding: utf-8 -*-

# regressione lineare mediante funzione di sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

plt.style.use('fivethirtyeight')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'

# legge i dati in dataframe pandas
data = pd.read_csv("mammals.csv", delimiter=',')

# calcola dimensione dei dati
n = len(data)

# Estrae dal dataframe l'array X delle features e l'array t dei valori target
X = np.array(data['body']).reshape(-1, 1)
t = np.array(data['brain']).reshape(-1, 1)

# visualizza dati mediante scatter
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.scatter(data['body'], data['brain'], s=20,c='C0', marker='o', alpha=.5)
plt.xlabel(u'Peso', fontsize=10)
plt.ylabel('Peso del cervello', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# definisce un modello sklearn di regressione lineare
r = linear_model.LinearRegression(fit_intercept=True)
# applica il modello sul training set
r.fit(X, t)
w=[np.asscalar(r.intercept_), np.asscalar(r.coef_)]
print (r.coef_, r.intercept_)

# visualizza retta di regressione sui dati
lw=min(data['body'])-np.abs(min(data['body'])*.2)
up=max(data['body'])+np.abs(max(data['body'])*.2)
t1 = np.arange(lw,up, 0.1)
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('white')
ax.scatter(data['body'], data['brain'], s=20,c='C0', marker='o', alpha=.8)
ax.plot(t1, w[0]+w[1]*t1, c='C1', linewidth=0.7, alpha=1)
plt.xlabel(u'Peso', fontsize=10)
plt.ylabel(r'Dimensione cervello', fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()