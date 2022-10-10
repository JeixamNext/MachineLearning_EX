import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("ciudades.csv", sep=";") #index_col=0
print(df.head(5))
print('Cantidad de Filas y columnas:',df.shape)
print('Nombre columnas:',df.columns)
df.info()
df.describe()
print("matriz de correlacion")
corr = df.set_index('alpha_3').corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()
# Aqui vemos la población año tras año de España

df_pop_es = df_pop[df_pop["country"] == 'Spain' ]
df_pop_es.head()

df_pop_es.drop(['country'],axis=1)['population'].plot(kind='bar')

