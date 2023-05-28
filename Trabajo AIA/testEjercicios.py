import funciones
import numpy as np
import datos.votos as votos


def testFuncion1():
    test = round(np.random.random(),3)
    conjunto_train,conjunto_test= funciones.ejercicio1(votos.datos, votos.clasif, test)
    proporcion = round(len(conjunto_test)/len(conjunto_train+conjunto_test),3)
    print("Proporcion parametro {} \nTamanyo train: {} \nTamanyo test: {} \nProporcion real {}".format(test,len(conjunto_train),len(conjunto_test),proporcion))


    
testFuncion1()