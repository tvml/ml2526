# -*- coding: utf-8 -*-

# regressione lineare con gradiente e sperimentazione per vari valori del coefficiente alpha
# nel gradiente
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# definisce un vettore di colori
colors = sns.color_palette("husl", 6)
# dichiara alcune proprietà grafiche della figura
sns.set(style="darkgrid", context='paper', palette=colors, rc={"figure.figsize": (16, 8),'image.cmap': 'jet', 'lines.linewidth':1})


def normalizza(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    return (X-mu)/sigma
    
# legge i dati in dataframe pandas
data = pd.read_csv("mammals.csv", delimiter=',')

# calcola dimensione dei dati
n = len(data)

# visualizza dati mediante scatter
fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.gca()
ax.scatter(data['body'], data['brain'], s=20,c='r', marker='o', alpha=.5)
plt.xlabel(u'Peso corporeo', fontsize=14)
plt.ylabel('Peso del cervello', fontsize=14)
plt.show()

# Regressione mediante discesa del gradiente on-line, una dimensione alla volta
# Estrae dal dataframe gli array X delle features e y dei valori
X, t = np.array(data[['body']]), np.array(data['brain']).reshape((-1,1))

X=normalizza(X)
X = np.column_stack((np.ones(n), X))

# mostra distribuzione dell'errore quadratico medio al variare dei coefficienti

# insieme dei valori considerati per i coefficienti
w0_list = np.linspace(80, 160, 100)
w1_list = np.linspace(80, 160, 100)

# crea una griglia di coppie di valori 
w0, w1 = np.meshgrid(w0_list, w1_list)

# definisce la funzione da calcolare in ogni punto della griglia
def error(v1, v2):
    theta = np.array((v1, v2)).reshape(-1, 1)
    e=(np.dot(X,theta)-t)
    return np.dot(e.T,e)[0,0]/(2*n)

v_error=np.vectorize(error)
e=v_error(w0,w1).T

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.gca(projection='3d')
ax.plot_surface(w0, w1, e, rstride=1, cstride=1, cmap=plt.cm.jet , linewidth=0, antialiased=False)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)
plt.xlabel(r"$\theta_0$", fontsize=12)
plt.ylabel(r"$\theta_1$", fontsize=12)
plt.title(r"Errore quadratico medio al variare dei coefficienti $w_0,w_1$", fontsize=12)
plt.show()

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.gca(projection='3d')
ax.contour(w0, w1, e, np.logspace(3.5,4.2,400), colors=colors, linewidth=1)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
ax.tick_params(axis='z', labelsize=8)
plt.xlabel(r"$w_0$", fontsize=12)
plt.ylabel(r"$w_1$", fontsize=12)
plt.title(r"Errore quadratico medio al variare dei coefficienti $w_0,w_1$", fontsize=12)
plt.show()

fig = plt.figure(figsize=(12,12))
fig.patch.set_facecolor('white')
ax = fig.gca()
im = plt.imshow(e, origin='lower', extent=(w0_list.min(),w0_list.max(),w1_list.min(), w1_list.max()), aspect='auto',alpha=.8)
plt.contour(w0, w1, e,color='r', lw=0.7)
ax.tick_params(axis='x', labelsize=8)
ax.tick_params(axis='y', labelsize=8)
plt.xlabel(r"$w_0$", fontsize=12)
plt.ylabel(r"$w_1$", fontsize=12)
plt.title(r"Errore quadratico medio al variare dei coefficienti $w_0,w_1$", fontsize=12)
fig.colorbar(im, shrink=0.5, aspect=7, cmap=plt.cm.jet)
plt.show()


# Regressione mediante discesa del gradiente
# memorizza la sequenza dei costi
def gradiente( X, t, w_init, alpha, iterations, n):
    w = np.copy(w_init)
    costs = []
    for i in range(iterations):
        e=np.dot(X,w)-t
        c=np.dot(e.T,e)[0,0]/(2*n)
        dw=np.dot(X.T, e)/n
        w = w-alpha*dw
        costs.append(c)
    return costs, w
    
# Inizializza il vettore dei coefficienti
w_init = np.array([0, 0]).reshape(-1,1)
    
iterations = 50
alphas = [0.01, 0.03, 0.1, 0.3, 0.5, 1]

number_of_iterations = range(1, iterations + 1)

fig = plt.figure()
fig.patch.set_facecolor('white')
for c,alpha in enumerate(alphas):
    costs, w = gradiente(X, t, w_init, alpha, iterations, n)
    ax = fig.add_subplot(3, 2, c+1, axisbg="#F8F8F8")
    ax = fig.gca()
    ax.plot(number_of_iterations, costs, c= colors[c], alpha=.5, lw=1.5)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel(r'Numero di iterazioni', fontsize=8)
    plt.ylabel(r'Costo', fontsize=8)
    plt.title(r'$\alpha$='+str(alpha)+r': $c$='+str(round(costs[-1],3))+r'; $w_0$='+str(round(w[0][0],2))+'; $w_1$='+str(round(w[1][0],2))+'', fontsize=10)
plt.tight_layout()
plt.show()

