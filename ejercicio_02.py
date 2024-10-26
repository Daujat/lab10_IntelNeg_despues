import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data_vino.csv', sep=';')

#variables predictoras y la variable objetivo
X = data[['Magnesium', 'Total_phenols', 'Color_intensity', 'Nonflavanoid_phenols']]
y = data['tipo']

#datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

#modelo XGBoost
model = XGBClassifier(random_state=42)

#entrenar modelo
model.fit(X_train, y_train)

#predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

#calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)