
import pandas as pd
import numpy as np

from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
  
from itertools import permutations 
  
def find_moda(datos):
    # Recuerda que este arreglo puede ser llenado por un usuario: https://parzibyte.me/blog/2020/10/18/python-llenar-arreglo-datos-usuario/
    #datos = [9, 2, 3, 4, 4, 4, 5, 6, 7, 8, 4, 9, 4]
    diccionario_conteo = {}
    for numero in datos:
        clave = str(numero)
        # Si no existe...
        if not clave in diccionario_conteo:
            # lo agregamos:
            diccionario_conteo[clave] = 1
        # Si ya existe...
        else:
            # Lo aumentamos
            diccionario_conteo[clave] += 1

    # Ahora recorremos el diccionario y obtenemos el mayor. Vamos a buscar el que tenga la mayor frecuencia
    frecuencia_mayor = 0
    numero_mas_repetido = datos[0]
    # Imprimimos el diccionario solo para depurar
    print(diccionario_conteo)
    # Y sacamos el mayor
    for numero in diccionario_conteo:
        if diccionario_conteo[numero] > frecuencia_mayor:
            numero_mas_repetido = numero
            frecuencia_mayor = diccionario_conteo[numero]
    # Finalmente imprimimos el más repetido, con su conteo
    conteo = diccionario_conteo[str(numero_mas_repetido)]
    print(f"El número que más se repite es {numero_mas_repetido} (encontrado {conteo} ocasiones)" )
    return numero_mas_repetido
  

#region DATA SET AND CLEAN DATA
dataset = pd.read_csv('data/NCDB_1999_to_2014.csv')


print(len(dataset))
print(dataset)

dataset = dataset.drop('C_YEAR', 1)
dataset = dataset.drop('C_MNTH', 1)
dataset = dataset.drop('C_WDAY', 1)
dataset = dataset.drop('C_HOUR', 1)

dataset = dataset.drop('V_ID', 1)
dataset = dataset.drop('V_YEAR', 1)

dataset = dataset.drop('P_AGE', 1)
dataset = dataset.drop('P_ID', 1)
dataset = dataset.drop('P_SEX', 1)
dataset = dataset.drop(dataset[dataset.values=="U"].index )
dataset = dataset.drop(dataset[dataset.values=="X"].index )
dataset = dataset.drop(dataset[dataset.values=="Q"].index )
dataset = dataset.drop(dataset[dataset.values=="N"].index )

dataset = dataset.drop(dataset[dataset.values=="UU"].index )
dataset = dataset.drop(dataset[dataset.values=="XX"].index )
dataset = dataset.drop(dataset[dataset.values=="QQ"].index )
dataset = dataset.drop(dataset[dataset.values=="NN"].index )

dataset = dataset.drop(dataset[dataset.values=="UUUU"].index )
dataset = dataset.drop(dataset[dataset.values=="XXXX"].index )
dataset = dataset.drop(dataset[dataset.values=="QQQQ"].index )
dataset = dataset.drop(dataset[dataset.values=="NNNN"].index )

dataset = dataset.reset_index(drop=True)

dataset = dataset.astype(int)

print(len(dataset))
print(dataset)
#endregion


#region RANDOM TREE
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, -1].values


forest_data = []
boost_data = []

for i in list(range(1)):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2  , random_state=0 )

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    classificator = RandomForestClassifier(n_estimators=20 , random_state=0)
    forest = classificator.fit(X_train, y_train)
    y_pred = classificator.predict(X_test)
    
    moda = find_moda(datos=y_pred)
    print("RandomForest user more exposed :" , moda)
    forest_data.append(moda)

    print("CONFUSION MATTIX : \n", confusion_matrix(y_test,y_pred))
    print("CLASSIFICATION REPORT : \n",classification_report(y_test,y_pred))
    print("ACCURACY SCORE : \n", accuracy_score(y_test, y_pred))
    #endregion 

    #region XGBOOST
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    
    
    moda = find_moda(datos=y_pred)
    print("XG Boost user more exposed  :" , moda)
    boost_data.append(moda)

    print("CONFUSION MATTIX : \n", confusion_matrix(y_test,y_pred))
    print("CLASSIFICATION REPORT : \n",classification_report(y_test,y_pred))
    print("ACCURACY SCORE : \n", accuracy_score(y_test, y_pred))

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #endregion

print("RandomForest final :" , find_moda(datos=forest_data) )

print("XG Boost final :" , find_moda(datos=forest_data) )