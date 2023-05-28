import funciones
import numpy as np
import carga_datos as cd

def testFuncion1():
    test = round(np.random.random(),3)
    conjunto_train,conjunto_test= funciones.ejercicio1(cd.X_cancer, cd.y_cancer, test)
    print(conjunto_train[0])
    proporcion = round(len(conjunto_test)/len(conjunto_train+conjunto_test),3)
    print("Proporcion parametro: {} \nProporcion real: {}\nTamanyo train: {} \nTamanyo test: {}".format(test,len(conjunto_train),len(conjunto_test),proporcion))    

#testFuncion1()

def test2_1():
    pass

