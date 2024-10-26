import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data_vino.csv', sep=';')

#Seleccionar las variables predictoras y la variable objetivo
X = data[['Magnesium', 'Total_phenols', 'Color_intensity', 'Nonflavanoid_phenols']]
y = data['tipo']

#Calcular la matriz de correlaciones
correlation_matrix = X.corr()
print("Matriz de correlaciones:")
print(correlation_matrix)

#Eliminar variables con correlación fuera del rango -0.6 a 0.6
columns_to_remove = []
for column in X.columns:
    if (correlation_matrix[column].abs() > 0.6).any():
        columns_to_remove.append(column)

X = X.drop(columns=columns_to_remove)

print("\nVariables seleccionadas después de eliminar correlaciones fuertes:")
print(X.columns)

#Verificar si hay variables
if len(X.columns) == 0:
    print("No hay variables restantes después de eliminar las correlaciones fuertes.")
    print("No se puede entrenar el modelo.")
else:
    #Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

    #Crear el modelo XGBoost
    model = XGBClassifier(random_state=42)

    #Entrenar el modelo
    model.fit(X_train, y_train)

    #Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    #Calcular la precisión del modelo
    accuracy = accuracy_score(y_test, y_pred)
    print("\nPrecisión del modelo:", accuracy)