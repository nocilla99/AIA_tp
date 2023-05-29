import trabajo_aia_2022_2023 as main
import numpy as np
import carga_datos as cd

X = cd.X_votos
y= cd.y_votos
test = round(np.random.random(),3)

def test1():
    _,_,y_training,y_test= main.particion_entr_prueba(X, y, test)
    proporcion = round(len(y_test)/len(y_test+y_training),3)

    print("Proporcion parametro: {} \nProporcion real: {}\nTamanyo train: {} \nTamanyo test: {}".format(test,proporcion,len(y_training),len(y_test)))    
    print("Tamaño del conjunto",len(y))

def test2_1():
    X_training,X_test,y_training,y_test = main.particion_entr_prueba(X, y, test)
    normst=main.NormalizadorStandard()
    normst.ajusta(X_training)
    Xtr_n=normst.normaliza(X_training)
    Xte_n=normst.normaliza(X_test)
    print("Media: {} \nDesv Tipic: {}".format(np.average(normst.media), np.average(normst.desv_tipica)))
    
    # #Caso para dar el raise
    # normst2=main.NormalizadorStandard()
    # Xte_n=normst2.normaliza(X_test)

test2_1()