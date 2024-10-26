import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data_vino.csv', sep=';')

# Seleccionar las variables predictoras y la variable objetivo
X = data[['Magnesium', 'Total_phenols', 'Color_intensity', 'Nonflavanoid_phenols']]
y = data['tipo']

#Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

print("Tamaño del conjunto de entrenamiento:", len(X_train))
print("Tamaño del conjunto de prueba:", len(X_test))