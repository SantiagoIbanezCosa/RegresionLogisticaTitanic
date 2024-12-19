import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def check_file_exists(file_path):
    """Verifica la existencia del archivo en la ruta especificada."""
    print(f'Verificando la existencia del archivo en la ruta: {file_path}')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'El archivo {file_path} no existe.')

def save_predictions(output_path, passenger_ids, predictions):
    """Guarda las predicciones en un archivo CSV."""
    output = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions})
    output.to_csv(output_path, index=False)
    print(f'Predicciones guardadas en {output_path}')
    print(output.head())

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    check_file_exists(file_path)
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocesa los datos eliminando columnas irrelevantes, rellenando valores nulos y convirtiendo variables categóricas."""
    # Eliminar columnas irrelevantes
    data = data.drop(columns=['Name', 'Ticket', 'Cabin'])
    
    # Rellenar valores nulos
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    
    # Convertir variables categóricas a numéricas
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
    
    return data

def train_model(X_train, y_train):
    """Entrena un modelo de regresión logística."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo utilizando precisión, matriz de confusión y reporte de clasificación."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(f'Precisión: {accuracy}')
    print('Matriz de confusión:')
    print(cm)
    print('Reporte de clasificación:')
    print(cr)
    return y_pred

def main():
    train_file_path = 'C:/Users/Santiago/Desktop/PruebasCienciaDeDatos/titanic/train.csv'
    test_file_path = 'C:/Users/Santiago/Desktop/PruebasCienciaDeDatos/titanic/test.csv'
    
    # Cargar y preprocesar datos de entrenamiento
    train_data = load_data(train_file_path)
    train_data = preprocess_data(train_data)
    
    # Separar características y objetivo
    features = train_data.drop(columns=['Survived', 'PassengerId'])
    target = train_data['Survived']
    
    # Dividir los datos en entrenamiento y validación
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # Entrenar el modelo
    model = train_model(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = evaluate_model(model, X_test, y_test)
    
    # Guardar el modelo
    joblib.dump(model, 'modelo_regresion_logistica_titanic.pkl')
    
    # Cargar y preprocesar datos de prueba
    test_data = load_data(test_file_path)
    test_data = preprocess_data(test_data)
    X_test_final = test_data.drop(columns=['PassengerId'])
    passenger_ids = test_data['PassengerId']
    
    # Realizar predicciones en el conjunto de prueba
    predictions = model.predict(X_test_final)
    
    # Guardar las predicciones
    save_predictions('C:/Users/Santiago/Desktop/PruebasCienciaDeDatos/titanic/predictions_logistic.csv', passenger_ids, predictions)

if __name__ == "__main__":
    main()