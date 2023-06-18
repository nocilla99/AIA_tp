import trabajo_aia_2022_2023 as main
import numpy as np
import carga_datos as cd
import pruebasdani as pd

X = cd.X_cancer
y= cd.y_cancer
test = round(np.random.random(),3)
def test1():
    print("--------------------------- TEST 1-----------------")
    _,_,y_training,y_test= main.particion_entr_prueba(X, y, test)
    proporcion = round(len(y_test)/len(np.concatenate([y_test,y_training])),1)
    print("Proporcion parametro: {} \nProporcion real: {}\nTamanyo train: {} \nTamanyo test: {}".format(test,proporcion,len(y_training),len(y_test)))    
    print("Tamaño del conjunto",len(y))
    print("--------------------------------------------")

def test2_1():
    print("--------------------------- TEST 2.1-----------------")
    X_training,X_test,y_training,y_test = main.particion_entr_prueba(X, y, test)
    normst=main.NormalizadorStandard()
    normst.ajusta(X_training)
    Xtr_n=normst.normaliza(X_training)
    Xte_n=normst.normaliza(X_test)
    print("Media: {} \nDesv Tipic: {}".format(np.average(normst.media), np.average(normst.desv_tipica)))
    
    # #Caso para dar el raise
    # normst2=main.NormalizadorStandard()
    # Xte_n=normst2.normaliza(X_test)
    print("--------------------------------------------")


def test2_2():
    print("--------------------------- TEST 2.2-----------------")
    X_training,X_test,y_training,y_test = main.particion_entr_prueba(X, y, test)
    normst=main.NormalizadorMinMax()
    normst.ajusta(X_training)
    Xtr_n=normst.normaliza(X_training)
    Xte_n=normst.normaliza(X_test)
    print("Minimos: {} \nMaximos: {}".format(np.average(normst.minimos), np.average(normst.maximos)))
    
    # #Caso para dar el raise
    # normst2=main.NormalizadorStandard()
    # Xte_n=normst2.normaliza(X_test)
    print("--------------------------------------------")

def test3():
    print("--------------------------- TEST 3-----------------")

    X= cd.X_votos
    y = cd.y_votos
    
    X_partir,X_test,y_partir,y_test = main.particion_entr_prueba(X, y, 0.3)
    X_training,X_vali,y_training,y_vali = main.particion_entr_prueba(X_partir,y_partir,0.3)

    clasif_rl = main.RegresionLogisticaMiniBatch(0.1,n_epochs=15,batch_tam=16)
    clasif_rl.entrena(X_training,y_training,X_vali,y_vali,40,True,True,10)
    _,y_test_pro = main.procesar_y(np.unique(y),y_test)
    tasa  = main.rendimiento(clasif_rl,X_test,y_test_pro)

    print("Rendimiento test:",tasa)
    print("--------------------------------------------")



def test4():
    
    print("--------------------------- TEST 4-----------------")
    print("--------------------------------------------")
    pass


def test7():

    print("--------------------------- TEST 7-----------------")
    Xc=np.array([["a",1,"c","x"],
                  ["b",2,"c","y"],
                  ["c",1,"d","x"],
                  ["a",2,"d","z"],
                  ["c",1,"e","y"],
                  ["c",2,"f","y"]])
    codifi_X = main.codifica_one_hot(Xc)

    print("La codificacion es: ",codifi_X)
    print("--------------------------------------------")

def test8_1():
    #######params###########
    rate = 0.1
    rate_decay = False 
    batch_tam = 64
    n_epochs = 50
    salida_epoch = False
    muestras_randoms = 5
    #########################
    print("--------------------------- TEST 8.1-----------------")
    X = cd.X_votos
    y = cd.y_votos
    X_tr,X_te,y_tr,y_te = main.particion_entr_prueba(X, y, 0.3)

    conjunto_codeado = main.codifica_one_hot(X_tr)


    clasif_0vr = main.RL_OvR(rate,rate_decay,batch_tam)
    clasif_0vr.entrena(X_tr,y_tr,n_epochs,salida_epoch)

    _, y_te_procesadas = main.procesar_y(np.unique(y_te),y_te)
    tasa = main.rendimiento(clasif_0vr,X_te,y_te)

    print("Rendimiento test:",tasa)
    for i in range(0,muestras_randoms):
        ejemplo_random = np.random.randint(0,len(X)-1)
        dato_ejemplo = X[ejemplo_random]
        print(f"*Se pretende clasificar el ejemplo con el indice {ejemplo_random}: {dato_ejemplo}")

        prediccion_ejemplo = clasif_0vr.clasifica([dato_ejemplo])
        print(f"{y[ejemplo_random] == prediccion_ejemplo[0]} -> Clasificacion real: {y[ejemplo_random]}; Predicción del modelo: {prediccion_ejemplo[0]}")
    
    print("--------------------------------------------")


def test_OP():
    
    print("--------------------------- TEST OP-----------------")
    #print("Codifi: ",main.codifica_one_hot(y))
    
    #print(X)
    #parm = {"batch_tam":8,"rate":0.1,"rate_decay":False}
    #rend_valid = main.rendimiento_validacion_cruzada(main.RL_Multinomial,parm,X,y,n=5)
    #print(rend_valid)
    X_training,X_test,y_training,y_test = main.particion_entr_prueba(X, y, 0.3)
   
    iris = main.RL_Multinomial(rate=0.01,batch_tam=64,rate_decay=False)
    iris.entrena(X_training, y_training,n_epochs=39, salida_epoch=True)
    rend = main.rendimiento(iris, X_training, y_training)
    print("Rendimiento: ", rend)
    print("--------------------------------------------")





# test1()
# test2_1()
# test2_2()
# test3()
# test4()
# test7()
# test8_1()
# test_OP()