import trabajo_aia_2022_2023 as main
import numpy as np
import carga_datos as cd
import pruebasdani as pd
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


def test2_2():
    X_training,X_test,y_training,y_test = main.particion_entr_prueba(X, y, test)
    normst=main.NormalizadorMinMax()
    normst.ajusta(X_training)
    Xtr_n=normst.normaliza(X_training)
    Xte_n=normst.normaliza(X_test)
    print("Minimos: {} \nMaximos: {}".format(np.average(normst.minimos), np.average(normst.maximos)))
    
    # #Caso para dar el raise
    # normst2=main.NormalizadorStandard()
    # Xte_n=normst2.normaliza(X_test)

def test3():
    cancer = main.RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)
    cancer.entrena(X=X,y=y,salida_epoch=True)
    rend = main.rendimiento(cancer,X,y)
    print("Rendimeinto :",rend)




def test4():
    X_training,X_test,y_training,y_test = main.particion_entr_prueba(X, y, 0.3)
    X_training,X_v = X_training[:len(X_training)-50] , X_training[len(X_training)-50:]
    y_training,y_v = y_training[:len(y_training)-50] , y_training[len(y_training)-50:]
    modelo = pd.RegresionLogisticaMiniBatch(0.05,n_epochs=100,batch_tam=16)
    modelo.entrena(X_training,y_training,None,None,50,True,True,3)

    # array_predicciones = modelo.clasifica(X_test)
    # array_predicciones = modelo.clasifica_prob(array_predicciones)

    # print(main.rendimiento(modelo,X_test,y_test))





def test7():

    Xc=np.array([["a",1,"c","x"],
                  ["b",2,"c","y"],
                  ["c",1,"d","x"],
                  ["a",2,"d","z"],
                  ["c",1,"e","y"],
                  ["c",2,"f","y"]])
    codifi_X = main.codifica_one_hot(Xc)

    print("La codificacion es: ",codifi_X)

def test8_1():

    codifica_x = main.codifica_one_hot(X)
    credito = main.RL_OvR(rate=0.1, rate_decay=False, batch_tam=64)
    credito.entrena(codifica_x,y, n_epochs=100,salida_epoch=False)

    nuev_ej = np.array([[0,1,0,0,1,1,0,0,1,0,1,0,0]])
    pred = credito.clasifica(nuev_ej)

    print("Predicciones: ",pred)

def test_OP():
    print("Codifi: ",main.codifica_one_hot(y))
    print(X)
    #parm = {"batch_tam":8,"rate":0.1,"rate_decay":False}
    #rend_valid = main.rendimiento_validacion_cruzada(main.RL_Multinomial,parm,X,y,n=5)
    #print(rend_valid)
    iris = main.RL_Multinomial(rate=0.1,batch_tam=8,rate_decay=False)
    iris.entrena(X, y,n_epochs=2, salida_epoch=True)
    rend = main.rendimiento(iris, X, y)
    print("Rendimiento: ", rend)





# test1()
# test2_1()
# test2_2()
test3()
# test4()
# test7()

# test_OP()