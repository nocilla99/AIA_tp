import numpy as np
from carga_datos import y_cancer, X_cancer

'''
bucle recorrer len(y) (que es el mismo de "X").
Separar los conjuntos en distintas clases no serviria porque dice que tiene que aparecer en el mismo orden (*)
Mirar la cantidad de conjunto "y_train[clase]" y de "y_test[clase]" para clasificar dicha fila Xy en el conjunto que corresponda train o test 
cuando acabe el bucle devolver X_train, x_test,y_train, y_test

Estratifiación: Mantener la proporcion en todas las clases
* Aleatoria: Tiene que mantenerse el orden en el que aparece en los datos de entrada, Entonces no se pueden elegir de forma random los indices de una clase y 
             barajarlos antes de separarlos en train y test no?

'''
def ejercicio1(X,y,test):
    