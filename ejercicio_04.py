import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

data = pd.read_csv('data_vino.csv', sep=';')

# Seleccionar las variables predictoras y la variable objetivo
X = data[['Magnesium', 'Nonflavanoid_phenols']]  # Variables seleccionadas previamente
y = data['tipo']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=42)

# Crear el modelo XGBoost
model = XGBClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# AUC
auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
print("AUC:", auc)