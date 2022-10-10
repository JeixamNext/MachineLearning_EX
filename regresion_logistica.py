import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline

dataframe = pd.read_csv(r"usuarios.csv")
dataframe.head()

dataframe.describe()

print(dataframe.groupby('clase').size())

# visualizar los datos
dataframe.drop(['clase'],1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='clase',size=4,vars=["duracion", "paginas","acciones","valor"],kind='reg')

# crear modelo

X = np.array(dataframe.drop(['clase'],1))
y = np.array(dataframe['clase'])
X.shape
# prediccion 
predictions = model.predict(X)
print(predictions[0:5])
model.score(X,y)
# reporte de resultados
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
