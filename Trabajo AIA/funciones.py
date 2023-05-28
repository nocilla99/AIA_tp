import numpy as np
from carga_datos import y_cancer, X_cancer
#########################Ejercicio1
'''
bucle recorrer len(y) (que es el mismo de "X").
Separar los conjuntos en distintas clases no serviria porque dice que tiene que aparecer en el mismo orden (*)
Mirar la cantidad de conjunto "y_train[clase]" y de "y_test[clase]" para clasificar dicha fila Xy en el conjunto que corresponda train o test 
cuando acabe el bucle devolver X_train, x_test,y_train, y_test

Estratifiación: Mantener la proporcion en todas las clases
* Aleatoria: Tiene que mantenerse el orden en el que aparece en los datos de entrada, Entonces no se pueden elegir de forma random los indices de una clase y 
             barajarlos antes de separarlos en train y test no?

'''
y = y_cancer
X = X_cancer

'''
Aqui se indica qué fila pertenece a cada clase. 
Se crea un diccionario donde las keys son las clases y los valores de cada key son las filas de esa clase

Entrada: la lista de clasificaciones
Salida: indices de las filas de cada clase
'''
def separar_indices(y):
    diccionario_indices= dict()

    for i in range(0,len(y)):
        if y[i] in diccionario_indices:
            diccionario_indices[y[i]].append(i)
        else:
            diccionario_indices[y[i]]=[i]

    #print (diccionario_indices)
    return diccionario_indices


'''
Shufflear los elementos de forma random, separarlos segun el atributo test(proporcion)
meterlos en listas de conjunto_training y conjunto_test
Ordenar cada conjunto en orden (para que se mantenga)

Entrada: separarIndices, test
Salida: indices de las filas para test e indices para training
'''

def separar_clases_test_training(y,test):
    indices_training = list()
    indices_test = list()
  
    diccionario_indices = separar_indices(y)

    for clase in diccionario_indices:
        np.random.shuffle(diccionario_indices[clase])  
        indices = len(diccionario_indices[clase])
        tamanyo = int(indices*test)
        indices_training += diccionario_indices[clase][:tamanyo-1]
        indices_test += diccionario_indices[clase][tamanyo-1:]
     
    
    indices_test.sort()
    indices_training.sort()
    # print(indices_training , indices_test)
    return indices_training,indices_test
        

'''
Entrada X, y , test(proporcion)
Separa los ejemplos X en conjunto de test y entrenamientos

Salida: conjunto_training,conjunto_test
'''
def separar_ejemplos(X,y,test):
    ind_tr, ind_te = separar_clases_test_training(y, test)
    conjunto_training = [ X[i] for i in ind_tr]
    conjunto_test = [ X[i] for i in ind_te]

    print(X[0])
    print(y[0])
    return conjunto_training,conjunto_test


######################### Fin Ejercicio1

#########################Ejercicio2
