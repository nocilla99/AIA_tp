#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
# ===================================================================
# Ampliación de Inteligencia Artificial, 2022-23
# PARTE I del trabajo práctico: Implementación de regresión logística
# Dpto. de CC. de la Computación e I.A. (Univ. de Sevilla)
# ===================================================================


# --------------------------------------------------------------------------
# Autor(a) del trabajo:
#
# APELLIDOS: Guillen Perez
# NOMBRE: Daniel
#
# Segundo(a) componente (si se trata de un grupo):
#
# APELLIDOS: Zaza
# NOMBRE: Zena
# ----------------------------------------------------------------------------


# ****************************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen. La discusión 
# y el intercambio de información de carácter general con los compañeros se permite, 
# pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de terceros, OBTENIDO A TRAVÉS
# DE LA RED o cualquier otro medio, se considerará plagio. En particular no se 
# permiten implementaciones obtenidas con HERRAMIENTAS DE GENERACIÓN AUTOMÁTICA DE CÓDIGO. 
# Si tienen dificultades para realizar el ejercicio, consulten con el profesor. 
# En caso de detectarse plagio (previamente con aplicaciones anti-plagio o durante 
# la defensa, si no se demuestra la autoría mediante explicaciones convincentes), 
# supondrá una CALIFICACIÓN DE CERO en la asignatura, para todos los alumnos involucrados. 
# Sin perjuicio de las medidas disciplinarias que se pudieran tomar. 
# *****************************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES, MÉTODOS
# Y ATRIBUTOS QUE SE PIDEN. EN PARTICULAR: NO HACERLO EN UN NOTEBOOK.

# NOTAS: 
# * En este trabajo NO SE PERMITE usar Scikit Learn (excepto las funciones que
#   se usan en carga_datos.py). 

# * SE RECOMIENDA y SE VALORA especialmente usar numpy. Las implementaciones 
#   saldrán mucho más cortas y eficientes, y se puntuarÁn mejor.   

import numpy as np

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aia.tgz y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn, por lo que para que la carga se haga sin
# problemas, deberá estar instalado el módulo sklearn). Todos los datos se
# cargan en arrays de numpy:

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresita (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn, con la opción
#   binary=True. Como vocabulario, se han usado las 609 palabras que ocurren
#   más frecuentemente en las distintas críticas. La vectorización binaria
#   convierte cada texto en un vector de 0s y 1s en la que cada componente indica
#   si el correspondiente término del vocabulario ocurre (1) o no ocurre (0)
#   en el texto (ver detalles en el archivo carga_datos.py). Los datos se
#   cargan finalmente en las variables X_train_imdb, X_test_imdb, y_train_imdb,
#   y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 



# ==================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA 
# ==================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test. La división ha de ser ALEATORIA y
# ESTRATIFICADA respecto del valor de clasificación. Por supuesto, en el orden 
# en el que los datos y los valores de clasificación respectivos aparecen en
# cada partición debe ser consistente con el orden original en X e y.   

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# >>> Xe_votos,Xp_votos,ye_votos,yp_votos=particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# >>> y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
#    (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# >>> np.unique(y_votos,return_counts=True)
#  (array([0, 1]), array([168, 267]))
# >>> np.unique(ye_votos,return_counts=True)
#  (array([0, 1]), array([112, 178]))
# >>> np.unique(yp_votos,return_counts=True)
#  (array([0, 1]), array([56, 89]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con los datos del cáncer, en el que se observa que las proporciones
# entre clases se conservan en la partición. 
    
# >>> Xev_cancer,Xp_cancer,yev_cancer,yp_cancer=particion_entr_prueba(X_cancer,y_cancer,test=0.2)

# >>> np.unique(y_cancer,return_counts=True)
# (array([0, 1]), array([212, 357]))

# >>> np.unique(yev_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yp_cancer,return_counts=True)
# (array([0, 1]), array([42, 71]))    


# Podemos ahora separar Xev_cancer, yev_cancer, en datos para entrenamiento y en 
# datos para validación.

# >>> Xe_cancer,Xv_cancer,ye_cancer,yv_cancer=particion_entr_prueba(Xev_cancer,yev_cancer,test=0.2)

# >>> np.unique(ye_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))

# >>> np.unique(yv_cancer,return_counts=True)
# (array([0, 1]), array([170, 286]))


# Otro ejemplo con más de dos clases:

# >>> Xe_credito,Xp_credito,ye_credito,yp_credito=particion_entr_prueba(X_credito,y_credito,test=0.4)

# >>> np.unique(y_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([202, 228, 220]))

# >>> np.unique(ye_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([121, 137, 132]))

# >>> np.unique(yp_credito,return_counts=True)
# (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#  array([81, 91, 88]))
# ------------------------------------------------------------------


## ---------- 
def particion_entr_prueba(X,y,test=0.2):
    
    clases, n_clases = np.unique(y, return_counts=True)
    tam_test = np.floor(n_clases * test).astype(int)
    tam_train = n_clases - tam_test

    
    indices = np.arange(len(y))
    indices_train = list()
    indices_test = list()

    for clase, n_train, n_test in zip(clases, tam_train, tam_test):
        indice = indices[y == clase]
        np.random.permutation(indice)
        indices_train.extend(indice[:n_train])
        indices_test.extend(indice[n_train:n_train + n_test])

    indices_train = np.array(indices_train)
    indices_test = np.array(indices_test)

    X_training, X_test = X[indices_train],X[indices_test]
    y_training, y_test = y[indices_train],y[indices_test]

    return X_training,X_test,y_training,y_test


#TEST
import carga_datos as cd
def test1():
    
    X = cd.X_cancer
    y= cd.y_cancer
    test = 0.3
    
    print("--------------------------- TEST 1 particiones -----------------")
    _,_,y_training,y_test= particion_entr_prueba(X, y, test)
    proporcion = round(len(y_test)/len(np.concatenate([y_test,y_training])),1)
    print("Proporcion parametro: {} \nProporcion real: {}\nTamanyo train: {} \nTamanyo test: {}".format(test,proporcion,len(y_training),len(y_test)))    
    print("Tamaño del conjunto",len(y))
    print("--------------------------------------------")


# ===========================
# EJERCICIO 2: NORMALIZADORES
# ===========================

# En esta sección vamos a definir dos maneras de normalizar los datos. De manera 
# similar a como está diseñado en scikit-learn, definiremos un normalizador mediante
# una clase con un metodo "ajusta" (fit) y otro método "normaliza" (transform).


# ---------------------------
# 2.1) Normalizador standard
# ---------------------------

# Definir la siguiente clase que implemente la normalización "standard", es 
# decir aquella que traslada y escala cada característica para que tenga
# media 0 y desviación típica 1. 

# En particular, definir la clase: 


# class NormalizadorStandard():

#    def __init__(self):

#         .....
        
#     def ajusta(self,X):

#         .....        

#     def normaliza(self,X):

#         ......

# 


# donde el método ajusta calcula las corresondientes medias y desviaciones típicas
# de las características de X necesarias para la normalización, y el método 
# normaliza devuelve el correspondiente conjunto de datos normalizados. 

# Si se llama al método de normalización antes de ajustar el normalizador, se
# debe devolver (con raise) una excepción:

class NormalizadorNoAjustado(Exception): pass


# Por ejemplo:
    
    
# >>> normst_cancer=NormalizadorStandard()
# >>> normst_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_n=normst_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_n=normst_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_n=normst_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, la media y desviación típica de Xe_cancer_n deben ser 
# 0 y 1, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_n, 
# ni con Xp_cancer_n. 



# ------ 

class NormalizadorStandard():

    def __init__(self):
        self.media = None
        self.desv_tipica = None
        
    def ajusta(self,X):
        #axis 0 se usa para indicar que lo que queremos normalizar es el atributo (columna)
        self.media= np.mean(X,axis=0)
        self.desv_tipica= np.std(X,axis=0)

    def normaliza(self,X):
        #Para lo del error: en el metodo anterior se le asigna un valor, entonces si no 
        #se ha ajustado antes el conjunto de datos la media y la desv seran None (el valor de la clase)
        if (self.media is None or self.desv_tipica is None):
            raise NormalizadorNoAjustado("Error, hay que ajustar antes de normalizar")
        
        res = (X-self.media)/self.desv_tipica
        return res


#TEST

def test2_1():
        
    X = cd.X_cancer
    y= cd.y_cancer
    test = round(np.random.uniform(0.2,0.8),3)

    print("--------------------------- TEST 2.1 norm standar-----------------")
    X_training,X_test,y_training,y_test = particion_entr_prueba(X, y, test)
    normst=NormalizadorStandard()
    normst.ajusta(X_training)
    Xtr_n=normst.normaliza(X_training)
    Xte_n=normst.normaliza(X_test)
    print("Media: {} \nDesv Tipic: {}".format(np.average(normst.media), np.average(normst.desv_tipica)))
    
    # #Caso para dar el raise
    # normst2=NormalizadorStandard()
    # Xte_n=normst2.normaliza(X_test)
    print("--------------------------------------------")

# ------------------------
# 2.2) Normalizador MinMax
# ------------------------

# Hay otro tipo de normalizador, que consiste en asegurarse de que todas las
# características se desplazan y se escalan de manera que cada valor queda entre 0 y 1. 
# Es lo que se conoce como escalado MinMax

# Se pide definir la clase NormalizadorMinMax, de manera similar al normalizador 
# del apartado anterior, pero ahora implementando el escalado MinMax.

# Ejemplo:

# >>> normminmax_cancer=NormalizadorMinMax()
# >>> normminmax_cancer.ajusta(Xe_cancer)
# >>> Xe_cancer_m=normminmax_cancer.normaliza(Xe_cancer)
# >>> Xv_cancer_m=normminmax_cancer.normaliza(Xv_cancer)
# >>> Xp_cancer_m=normminmax_cancer.normaliza(Xp_cancer)

# Una vez realizado esto, los máximos y mínimos de las columnas de Xe_cancer_m
#  deben ser 1 y 0, respectivamente. No necesariamente ocurre lo mismo con Xv_cancer_m,
# ni con Xp_cancer_m. 


# ------ 

class NormalizadorMinMax():

    def __init__(self):

        self.minimos = list()
        self.maximos = list()
        
    def ajusta(self,X):
    #Calcular el minimo y maximo de cada atributo y meterlas en un array 
        self.minimos = np.min(X,axis=0) 
        self.maximos = np.max(X,axis=0) 

    def normaliza(self,X):
        
        if (len(self.maximos) == 0):
            raise NormalizadorNoAjustado("Error, hay que ajustar antes de normalizar")
        
        '''
        Hacer las proporciones [(datos - min) / (max - min)]
        Devolver de nuevo el conjunto normalizado'''

        self.minimos = np.min(X,axis=0) 
        self.maximos = np.max(X,axis=0) 
        res = (X-self.minimos)/(self.maximos-self.minimos)
        return res

#TEST
def test2_2():
    
    X = cd.X_cancer
    y= cd.y_cancer
    test = round(np.random.uniform(0.2,0.8),3)
    print("--------------------------- TEST 2.2 norm minmax-----------------")
    X_training,X_test,y_training,y_test = particion_entr_prueba(X, y, test)
    normst= NormalizadorMinMax()
    normst.ajusta(X_training)
    normst.normaliza(X_training)
    print("Minimos: {} \nMaximos: {}".format(normst.minimos, normst.maximos))
    
    # #Caso para dar el raise
    # normst2=NormalizadorStandard()
    # Xte_n=normst2.normaliza(X_test)
    print("--------------------------------------------")


# ===========================================
# EJERCICIO 3: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# En este ejercicio se propone la implementación de un clasificador lineal 
# binario basado regresión logística (mini-batch), con algoritmo de entrenamiento 
# de descenso por el gradiente mini-batch (para minimizar la entropía cruzada).


# En concreto se pide implementar una clase: 

# class RegresionLogisticaMiniBatch():

#    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,
#                 batch_tam=64):

#         .....
        
#     def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False,
#                     early_stopping=False,paciencia=3):

#         .....        

#     def clasifica_prob(self,ejemplos):

#         ......
    
#     def clasifica(self,ejemplo):
                        
#          ......



# * El constructor tiene los siguientes argumentos de entrada:



#   + rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#     durante todo el aprendizaje. Si rate_decay es True, rate es la
#     tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#   + rate_decay, indica si la tasa de aprendizaje debe disminuir en
#     cada epoch. En concreto, si rate_decay es True, la tasa de
#     aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#     con la siguiente fórmula: 
#        rate_n= (rate_0)*(1/(1+n)) 
#     donde n es el número de epoch, y rate_0 es la cantidad introducida
#     en el parámetro rate anterior. Su valor por defecto es False. 
#  
#   + batch_tam: tamaño de minibatch


# * El método entrena tiene como argumentos de entrada:
#   
#     +  Dos arrays numpy X e y, con los datos del conjunto de entrenamiento 
#        y su clasificación esperada, respectivamente. Las dos clases del problema 
#        son las que aparecen en el array y, y se deben almacenar en un atributo 
#        self.clases en una lista. La clase que se considera positiva es la que 
#        aparece en segundo lugar en esa lista.
#     
#     + Otros dos arrays Xv,yv, con los datos del conjunto de  validación, que se 
#       usarán en el caso de activar el parámetro early_stopping. Si son None (valor 
#       por defecto), se supone que en el caso de que early_stopping se active, se 
#       consideraría que Xv e yv son resp. X e y.

#     + n_epochs es el número máximo de epochs en el entrenamiento. 

#     + salida_epoch (False por defecto). Si es True, al inicio y durante el 
#       entrenamiento, cada epoch se imprime  el valor de la entropía cruzada 
#       del modelo respecto del conjunto de entrenamiento, y su rendimiento 
#       (proporción de aciertos). Igualmente para el conjunto de validación, si lo
#       hubiera. Esta opción puede ser útil para comprobar 
#       si el entrenamiento  efectivamente está haciendo descender la entropía
#       cruzada del modelo (recordemos que el objetivo del entrenamiento es 
#       encontrar los pesos que minimizan la entropía cruzada), y está haciendo 
#       subir el rendimiento.
# 
#     + early_stopping (booleano, False por defecto) y paciencia (entero, 3 por defecto).
#       Si early_stopping es True, dejará de entrenar cuando lleve un número de
#       epochs igual a paciencia sin disminuir la menor entropía conseguida hasta el momento
#       en el conjunto de validación 
#       NOTA: esto se suele hacer con mecanismo de  "callback" para recuperar el mejor modelo, 
#             pero por simplificar implementaremos esta versión más sencilla.  
#        



# * Método clasifica: recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY de clases que el modelo predice para esos ejemplos. 

# * Un método clasifica_prob, que recibe UN ARRAY de ejemplos (array numpy) y
#   devuelve el ARRAY con las probabilidades que el modelo 
#   asigna a cada ejemplo de pertenecer a la clase positiva.       
    

# Si se llama a los métodos de clasificación antes de entrenar el modelo, se
# debe devolver (con raise) una excepción:

class ClasificadorNoEntrenado(Exception): pass

        
  

# RECOMENDACIONES: 


# + IMPORTANTE: Siempre que se pueda, tratar de evitar bucles for para recorrer 
#   los datos, usando en su lugar funciones de numpy. La diferencia en eficiencia
#   es muy grande. 

# + Téngase en cuenta que el cálculo de la entropía cruzada no es necesario
#   para el entrenamiento, aunque si salida_epoch o early_stopping es True,
#   entonces si es necesario su cálculo. Tenerlo en cuenta para no calcularla
#   cuando no sea necesario.     

# * Definir la función sigmoide usando la función expit de scipy.special, 
#   para evitar "warnings" por "overflow":

from scipy.special import expit    

def sigmoide(x):
    return expit(x)

# * Usar np.where para definir la entropía cruzada. 

# -------------------------------------------------------------

# Ejemplo, usando los datos del cáncer de mama (los resultados pueden variar):


# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer,yv_cancer)

# >>> lr_cancer.clasifica(Xp_cancer_n[24:27])
# array([0, 1, 0])   # Predicción para los ejemplos 24,25 y 26 

# >>> yp_cancer[24:27]
# array([0, 1, 0])   # La predicción anterior coincide con los valores esperado para esos ejemplos

# >>> lr_cancer.clasifica_prob(Xp_cancer_n[24:27])
# array([7.44297196e-17, 9.99999477e-01, 1.98547117e-18])

# Para calcular el rendimiento de un clasificador sobre un conjunto de ejemplos, usar la 
# siguiente función:
    
def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]

# Por ejemplo, los rendimientos sobre los datos (normalizados) del cáncer:
    
# >>> rendimiento(lr_cancer,Xe_cancer_n,ye_cancer)
# 0.9824561403508771

# >>> rendimiento(lr_cancer,Xp_cancer_n,yp_cancer)
# 0.9734513274336283

# Ejemplo con salida_epoch y early_stopping:

# >>> lr_cancer=RegresionLogisticaMiniBatch(rate=0.1,rate_decay=True)

# >>> lr_cancer.entrena(Xe_cancer_n,ye_cancer,Xv_cancer_n,yv_cancer,salida_epoch=True,early_stopping=True)

# Inicialmente, en entrenamiento EC: 155.686323940485, rendimiento: 0.873972602739726.
# Inicialmente, en validación    EC: 43.38533009881579, rendimiento: 0.8461538461538461.
# Epoch 1, en entrenamiento EC: 32.7750241863029, rendimiento: 0.9753424657534246.
#          en validación    EC: 8.4952918658522,  rendimiento: 0.978021978021978.
# Epoch 2, en entrenamiento EC: 28.0583715052223, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.665719133490596, rendimiento: 0.967032967032967.
# Epoch 3, en entrenamiento EC: 26.857182744289368, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.09511082759361, rendimiento: 0.978021978021978.
# Epoch 4, en entrenamiento EC: 26.120803184993328, rendimiento: 0.9780821917808219.
#          en validación    EC: 8.327991940213478, rendimiento: 0.967032967032967.
# Epoch 5, en entrenamiento EC: 25.66005010760342, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.376171724729662, rendimiento: 0.967032967032967.
# Epoch 6, en entrenamiento EC: 25.329200890122557, rendimiento: 0.9808219178082191.
#          en validación    EC: 8.408704771704937, rendimiento: 0.967032967032967.
# PARADA TEMPRANA

# Nótese que para en el epoch 6 ya que desde la entropía cruzada obtenida en el epoch 3 
# sobre el conjunto de validación, ésta no se ha mejorado. 


# -----------------------------------------------------------------

'''''
Algoritmo de descenso por el gradiente(version mini-batch): 

•Inicializar los pesos(w0,...,wn) aleatoriamente 
•Para cada epoch: 
    •Dividir aleatoriamente los ejemplos de entrenamiento en grupos de P ejemplos(mini-batches): 
    •Para cada mini-batch B actualizar cada los pesos: 
       wi ← wi + η*sum j∈B( [(y(j) − σ(w*x(j)))x_i(j)])

El proceso de entrenamiento del clasificador lineal basado en regresión logística (mini-batch) implica los siguientes pasos:

1 - Inicialización de los pesos y el sesgo (bias) del modelo.
2 - División aleatoria del conjunto de entrenamiento en mini-batches.
3 - Iteración sobre cada mini-batch y actualización de los pesos y bias utilizando el gradiente descendente 
    para minimizar la función de costo (la entropía cruzada).
4 - Repetición de los pasos 2 y 3 durante un número fijo de epochs.
5 - Evaluación del rendimiento del modelo en un conjunto de validación .

'''

class RegresionLogisticaMiniBatch():

    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,batch_tam=64):
        
        self.rate= rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.clases = list()
        self.weight = None
        # parametro importatnte permite al modelo ajustar el imbral de decision es decir se utiliza para
        # desplazar la funcion de decsicion hacia arriba o hacia abajo ...
        self.bias  = None
  

    # funcion auxliar para incilizar pesos aleatoriamente 
    # y los bias en 0 porque en principio no tiene prefrencia hacia una clase en particular
    def inicializar_pesos(self, n_carac):
        self.weight = np.random.uniform(low=-1, high=1,size=(n_carac,1))
        self.bias = 0
        return self.weight, self.bias

    def entropia_cruzada(self, X,y):
        
        # predicciones = sigmoide(np.dot(X,self.weight)+self.bias)

        prod_escalares = [np.dot(fila, self.weight)[0] + self.bias for fila in X]
        predicciones = sigmoide(prod_escalares)

        #suma de las entropias cruzadas de cada ejemplo x
        # La formula de la ec es (-y * log(pred) - (1 - y) * log(1 - pred)), pero piden usar where. Habra que hacer las medias de cuando "y[i]" valga 1. y cuando sea 0.
        array_entropias = [np.where(y[i]==1, -np.log(np.maximum(predicciones[i], 000.1)), -np.log(np.maximum(1-predicciones[i], 000.1))) for i in range(0,len(predicciones))]
        coste = np.sum(array_entropias) 
        return coste/len(array_entropias)
    
    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False, early_stopping=False,paciencia=3):

        self.clases, y = procesar_y(np.unique(y), y)
        self.n_epochs = n_epochs

        #normalizar datos entradas
        norm = NormalizadorStandard()
        norm.ajusta(X)
        X = norm.normaliza(X)
        

        if(Xv is None or yv is None):
            yv = y
            Xv = X
        else :    
            norm.ajusta(Xv)
            Xv = norm.normaliza(Xv)
            _, yv = procesar_y(np.unique(yv), yv)
        
        

        mejor_entropia = np.Infinity
        epochs_sin_mejora = 0
        rate = self.rate
        # INICIALIZA LOS PESOS
        self.weight , self.bias = self.inicializar_pesos(X.shape[1])
        
        #Separar en batchs  
        partes = self.batch_tam      
        X_partes = np.array_split(X, (len(X)/self.batch_tam))
        y_partes = np.array_split(y, (len(y)/self.batch_tam))

        for epoch in range(self.n_epochs):

            if(self.rate_decay):
                rate = (rate)*(1/( 1 + self.n_epochs))

            #parte de minibatch
            for i in range(len(self.weight)):
                
                for i in range(0, X.shape[0], self.batch_tam):
                
                    X_batch = X[i:i + self.batch_tam]
                    y_batch = y[i:i + self.batch_tam] 

                    #wi ← wi + η*sum j∈B( [(y(j) − σ(w*x(j)))x_i(j)])
                    z = np.dot(X_batch, self.weight) + self.bias
                    y_pred =  sigmoide(aux[0])
                    pesos = np.dot(X_batch.T, y_pred - y_batch)
                    suma_bias = np.sum(y_pred - y_batch)

                    self.weight += rate*pesos
                    self.bias -= suma_bias

            if(early_stopping or salida_epoch):

                ec_Xv = self.entropia_cruzada(Xv, yv)
                
                if(salida_epoch):
                    
                    ec_X = self.entropia_cruzada(X, y)
                    
                    rendimiento_X = self.rendimiento(X,y)
                    
                    rendimiento_Xv = self.rendimiento(Xv,yv)

                    print(f"Epoch {epoch +1}, en entrenamiento EC: {ec_X}, rendimiento: {rendimiento_X}")
                    print(f"         en validación    EC: {ec_Xv}, rendimiento: {rendimiento_Xv}")

                if(early_stopping): 
                    if ( ec_Xv > mejor_entropia):
                        epochs_sin_mejora += 1

                        if(epochs_sin_mejora>=paciencia):
                            print("~~~~~~~~~~PARADA TEMPRANA~~~~~~~~~~")
                            break
                    else:
                        mejor_entropia = ec_Xv
                        epochs_sin_mejora = 0
    

    def clasifica_prob(self,ejemplos):
        if self.weight is None or self.bias is None:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado.")
        
        # vector z que contiene la union lineal para cada ejemplo despues de multiplicarle con su peso y le suma el sesgo
        z = [np.dot(fila, self.weight)[0] + self.bias for fila in ejemplos]
        # despues de aplicarle el sigmoide al vector z nos sale las probalidades de prediccion de cada ejemplo X
        predicciones = sigmoide(z) 
        return predicciones


    def clasifica(self,ejemplo):
        probabilidad = self.clasifica_prob(ejemplo)
        # entonces ahora despues de obtener la probadlidad asignamos que si la prob >= 0.5 enonces su clasificacion 1
        # sino le asiganmos una clasificacion 0
        return np.where(probabilidad > 0.5, self.clases[1], self.clases[0])

    def rendimiento(self, X, y):
        predicciones = self.clasifica(X)
        aciertos = (predicciones == y).sum()
        precision = aciertos / y.shape[0]
        return precision

def procesar_y(clases,lista_y):
    # La clase que se considera positiva es la que 
    # aparece en segundo lugar en esa lista.
    transf_binaria = np.where(lista_y==clases[0],0,1)
    return [0,1], transf_binaria

#TEST

def test3():
    
    X = cd.X_cancer
    y= cd.y_cancer
    test = 0.3
    print("--------------------------- TEST 3 RL-binario-----------------")

    X= cd.X_votos
    y = cd.y_votos
    
    X_partir,X_test,y_partir,y_test = particion_entr_prueba(X, y,test)
    X_training,X_vali,y_training,y_vali = particion_entr_prueba(X_partir,y_partir,0.3)

    clasif_rl = RegresionLogisticaMiniBatch(0.1,n_epochs=15,batch_tam=16)
    clasif_rl.entrena(X_training,y_training,X_vali,y_vali,40,True,True,10)
    _,y_test_pro = procesar_y(np.unique(y),y_test)
    tasa  = rendimiento(clasif_rl,X_test,y_test_pro)

    print("Rendimiento test:",tasa)
    print("--------------------------------------------")


# ------------------------------------------------------------------------------

# =================================================
# EJERCICIO 4: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este jercicio puede servir para el ajuste de parámetros en los ejercicios posteriores, 
# pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1


# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador (como por ejemplo 
# la clase RegresionLogisticaMiniBatch). El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cancer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
#                                {"batch_tam":16,"rate":0.01,"rate_decay":True},
#                                 Xe_cancer_n,ye_cancer,n=5)

# Partición: 1. Rendimiento:0.9863013698630136
# Partición: 2. Rendimiento:0.958904109589041
# Partición: 3. Rendimiento:0.9863013698630136
# Partición: 4. Rendimiento:0.9726027397260274
# Partición: 5. Rendimiento:0.9315068493150684
# >>> 0.9671232876712328


# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones DEBEN SER ALEATORIAS Y ESTRATIFICADAS. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
# >>> lr16.entrena(Xe_cancer_n,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(lr16,Xp_cancer_n,yp_cancer)
# 0.9646017699115044

#------------------------------------------------------------------------------


def rendimiento_validacion_cruzada(clase_clasificador,params,X,y,Xv=None,yv=None,n=5):
    # hemos usado permutation porque no afecta al orden original del conjunto de datos sino crea un array nuevo y lo altera
    # permutamos los indices de los datos aleatoriamente y despues con np.array_split dividimos los indices perumtados 
    # en 'n' partes iguales  
    particiones = np.array_split(np.random.permutation(range(len(X))), n)
    # para almacenar el rendimiento de cada iteracion en la valid_cruz
    rends = []
    # iteramos cada particion i en range n partes
    for i in range(n):
        # conectamos los partes de indices de train que no sean iguales al particion actual i
        indices_entrenamiento = np.concatenate([particiones[j] for j in range(n) if j != i])
        # almacenamos los partes de los indices de valid que son los i particciones actuales
        indices_validacion = particiones[i]
        
        # en este paso obtenemos los datos de train con los indices de train
        X_training = X[indices_entrenamiento]
        y_training = y[indices_entrenamiento]

        # comprobamos si los daots de Xv o yv son None entonces son iguales a los datos de X_train y y_train
        # sino obtenemos los datos de Xv y yv con los indices de valid
        if Xv is None or yv is None:
            Xv = X_training
            yv = y_training
        else :
            Xv = X[indices_validacion]
            yv = y[indices_validacion]

        # entrenamos los datos de training utilizando cualquier clasificador que queremos
        clasificador = clase_clasificador(**params)
        clasificador.entrena(X_training,y_training)

        # evaluamos el rendimineto de los daots de validacion
        rend = rendimiento(clasificador,Xv, yv)
        rends.append(rend)
        print(f"Partición {i+1}. Rendimiento: {rend}")

    # rendimiento medio entre todos los rendimintos
    rend_med = np.mean(rends)
    print("Rendimiento medio: ",rend_med)
    return rend_med

#TEST
def test4():
        
    X = cd.X_cancer
    y= cd.y_cancer
    print("--------------------------- TEST 4 val_cruzada -----------------")
    rend = 0.0
    while (rend < 0.5):
        rend = rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,
                               {"batch_tam":16,"rate":0.01,"rate_decay":True},
                                X,y,n=5)
    X_partir, X_te, y_partir, y_te = particion_entr_prueba(X, y, 0.2) 
    lr16=RegresionLogisticaMiniBatch(batch_tam=16,rate=0.01,rate_decay=True)
    lr16.entrena(X_partir,y_partir)
    print("Rendiemtno mejor: ",rendimiento(lr16,X_te,y_te))
    print("--------------------------------------------")


# ===================================================
# EJERCICIO 5: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando la regeresión logística implementada en el ejercicio 2, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros (tasa, rate_decay, batch_tam) para mejorar el rendimiento 
# (no es necesario ser muy exhaustivo, tan solo probar algunas combinaciones). 
# Si se ha hecho el ejercicio 4, usar validación cruzada para el ajuste 
# (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos
# sobre un conjunto de prueba.     

# Mostrar también, para cada conjunto de datos, un ejemplo con salida_epoch, 
# en el que se vea cómo desciende la entropía cruzada y aumenta el 
# rendimiento durante un entrenamiento.     

# ----------------------------


# AQUI DEPENDE A LOS RESULTADOS DEL EJRICCIOS 4 HASTA QUE NO TENGAMOS BIEN NO PUEDO HACERLO




#TEST
def test5():
    print("--------------------------- TEST 5 ejemplos -----------------")
    print("CONJUNTO VOTOS")
    X = cd.X_votos
    y= cd.y_votos
    test = 0.
    
    X_partir, X_te, y_partir, y_te = particion_entr_prueba(X, y, test)   
    X_tr,X_v,y_tr,y_v = particion_entr_prueba(X_partir, y_partir, test)
    modelovotos = RegresionLogisticaMiniBatch(rate=0.01,rate_decay=False,n_epochs=100,batch_tam=64)
    modelovotos.entrena(X_tr,y_tr,X_v, y_v,100,True,True,3)
    _,y_test_pro = procesar_y(np.unique(y),y_te)
    tasa  = rendimiento(modelovotos,X_te,y_test_pro)
    print("Rendimiento test",tasa)
    
    print("\nCONJUNTO CANCER")
    X = cd.X_cancer
    y= cd.y_cancer
    test = 0.3
    
    X_partir, X_te, y_partir, y_te = particion_entr_prueba(X, y, test)   
    X_tr,X_v,y_tr,y_v = particion_entr_prueba(X_partir, y_partir, test)
    modelocancer = RegresionLogisticaMiniBatch(rate=0.001,rate_decay=True,n_epochs=100,batch_tam=16)
    modelocancer.entrena(X_tr,y_tr,X_v, y_v,100,True,True,3)
    tasa  = rendimiento(modelocancer,X_te,y_te)
    print("Rendimiento test",tasa)
    
    print("\nCONJUNTO IMDB")
    X = cd.X_train_imdb
    y = cd.y_train_imdb
    test = 0.3
    X_partir, X_te, y_partir, y_te = particion_entr_prueba(X, y, test)   
    X_tr,X_v,y_tr,y_v = particion_entr_prueba(X_partir, y_partir, test)
    modeloCríticas = RegresionLogisticaMiniBatch(rate=0.1,rate_decay=False,n_epochs=50,batch_tam=16)
    modeloCríticas.entrena(X_tr,y_tr,X_v, y_v,50,False,False,3)
    _,y_test_pro = procesar_y(np.unique(y),y_te)
    tasa  = rendimiento(modeloCríticas,X_te,y_test_pro)
    print("Rendimiento test",tasa)
    print("--------------------------------------------") 



# =====================================================
# EJERCICIO 6: CLASIFICACIÓN MULTICLASE CON ONE vs REST
# =====================================================

# Se pide implementar un algoritmo de regresión logística para problemas de
# clasificación en los que hay más de dos clases, usando  la técnica One vs Rest. 


#  Para ello, implementar una clase  RL_OvR con la siguiente estructura, y que 
#  implemente un clasificador OvR (one versus rest) usando como base el
#  clasificador binario RegresionLogisticaMiniBatch


# class RL_OvR():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica(self,ejemplos):

#        ......
            



#  Los parámetros de los métodos significan lo mismo que en el apartado
#  anterior, aunque ahora referido a cada uno de los k entrenamientos a 
#  realizar (donde k es el número de clases).
#  Por simplificar, supondremos que no hay conjunto de validación ni parada
#  temprana.  

 

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------
# >>> Xe_iris,Xp_iris,ye_iris,yp_iris=particion_entr_prueba(X_iris,y_iris)

# >>> rl_iris_ovr=RL_OvR(rate=0.001,batch_tam=8)

# >>> rl_iris_ovr.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris_ovr,Xe_iris,ye_iris)
# 0.8333333333333334

# >>> rendimiento(rl_iris_ovr,Xp_iris,yp_iris)
# >>> 0.9
# --------------------------------------------------------------------


class RL_OvR():

     def __init__(self,rate=0.1,rate_decay=False,
                   batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        # diccionario para almacenar los clasificadores binarios
        self.dic_clasifi = {}
    
     def inicializar_pesos(self, n_carac):
        # no he puesto self aqui porque cuando utliza la clase de RegresionLogisticaMiniBatch y de ello
        # llama a la funcion inicializar_pesos para obtener los pessos y las bias iniciales estos valores iniciales se asignan
        # directamente a los atributos weight y bias aqui
        weight = np.random.randn(n_carac,1)
        bias = 0
        return weight, bias

     def entrena(self,X,y,n_epochs=100,salida_epoch=False):
         self.clases = np.unique(y)

        # para cada clase se crea un clasificador binario utilizando la clase RegresionLogisticaMiniBatch
        # 'y' se conveirta en un victor binario donde los elementos que corresponde a la clase actual son 1 y el resto 0 
         for clase in self.clases:
            y_binario = np.where(y == clase, 1, 0)

            clasifi = RegresionLogisticaMiniBatch(rate=self.rate, rate_decay=self.rate_decay, batch_tam=self.batch_tam)
            clasifi.entrena(X,y_binario, n_epochs=n_epochs, salida_epoch=salida_epoch)

            # clave = la clse
            # valor = clasificador binario de esa clase
            self.dic_clasifi[clase] = clasifi


    # realizamos la clasificacion para los nuevos ejemplos
     def clasifica(self,ejemplos):

        y_predicion = []

        # para cada ejemplo cogemos la probalidad de partenencia a cada clase
        # cogemos la mayor probalidad y se almacena en la lista y_prediccion
        for ejemplo in ejemplos:
            max_prob = -1
            clase_predicido = None

            for clase, clasifi in self.dic_clasifi.items():
                prob = clasifi.clasifica_prob([ejemplo])[0]

                if prob > max_prob:
                    max_prob = prob
                    clase_predicido = clase

            y_predicion.append(clase_predicido)
        
        return y_predicion

    

            
# --------------------------------

#TEST
def test6():
    print("--------------------------- TEST 6 Ovr-----------------")
    X = cd.X_iris
    y= cd.y_iris
    test = 0.3

    X_tr,X_te,y_tr,y_te = particion_entr_prueba(X, y, test)
    modelo_OvR = RL_OvR(0.01,False,60)
    modelo_OvR.entrena(X_tr,y_tr,100,False)

    rend = rendimiento(modelo_OvR,X_te,y_te)
    print("Rendimiento test:",rend)
    print("--------------------------------------------")



# =================================
# EJERCICIO 7: CODIFICACIÓN ONE-HOT
# =================================


# Los conjuntos de datos en los que algunos atributos son categóricos (es decir,
# sus posibles valores no son numéricos, o aunque sean numéricos no hay una 
# relación natural de orden entre los valores) no se pueden usar directamente
# con los modelos de regresión logística, o con redes neuronales, por ejemplo.

# En ese caso es usual transformar previamente los datos usando la llamada
# "codificación one-hot". Básicamente, cada columna se reemplaza por k columnas
# en los que los valores psoibles son 0 o 1, y donde k es el número de posibles 
# valores del atributo. El valor i-ésimo del atributo se convierte en k valores
# (0 ...0 1 0 ...0 ) donde todas las posiciones son cero excepto la i-ésima.  

# Por ejemplo, si un atributo tiene tres posibles valores "a", "b" y "c", ese atributo 
# se reemplazaría por tres atributos binarios, con la siguiente codificación:
# "a" --> (1 0 0)
# "b" --> (0 1 0)
# "c" --> (0 0 1)    

# Definir una función:    
    
#     codifica_one_hot(X) 

# que recibe un conjunto de datos X (array de numpy) y devuelve un array de numpy
# resultante de aplicar la codificación one-hot a X.Por simplificar supondremos 
# que el array de entrada tiene todos sus atributos categóricos, y que por tanto 
# hay que codificarlos todos.

# Aplicar la función para obtener una codificación one-hot de los datos sobre
# concesión de prestamo bancario.     
  
# >>> Xc=np.array([["a",1,"c","x"],
#                  ["b",2,"c","y"],
#                  ["c",1,"d","x"],
#                  ["a",2,"d","z"],
#                  ["c",1,"e","y"],
#                  ["c",2,"f","y"]])
   
# >>> codifica_one_hot(Xc)
# 
# array([[1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.],
#        [0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.],
#        [0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0.],
#        [1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.],
#        [0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.]])

# En este ejemplo, cada columna del conjuto de datos original se transforma en:
#   * Columna 0 ---> Columnas 0,1,2
#   * Columna 1 ---> Columnas 3,4
#   * Columna 2 ---> Columnas 5,6,7,8
#   * Columna 3 ---> Columnas 9, 10,11     

    
  

# -------- 


def codifica_one_hot(X):

    unique_values = np.unique(X)
    num_filas = X.shape[0]
    num_columnas = len(unique_values)

    # creamos una matriz de ceros con las dimensiones num_filas(que son las mismas filas de X) 
    # y num_columnas (que son igual al numeros de elementos difrentes de X)
    codifica_X = np.zeros((num_filas,num_columnas))

    for i , value in enumerate(unique_values):
        # creamos m para saber las posiciones de cada valor en unique_values en X, es decir donde los elementos de X igual a value es True
        # y donde no son es False 
        m = X == value 
        # con m podemos seleccionar las filas en codifica_X y i para seleccionar las columnas en la que debemos asignar 1 en codifica_X
        codifica_X[np.where(m),i] = 1

    return codifica_X


#TEST
def test7():
    
    X = cd.X_cancer
    y= cd.y_cancer
    test = 0.3
    print("--------------------------- TEST 7 cod hot enc-----------------")
    Xc=np.array([["a",1,"c","x"],
                  ["b",2,"c","y"],
                  ["c",1,"d","x"],
                  ["a",2,"d","z"],
                  ["c",1,"e","y"],
                  ["c",2,"f","y"]])
    codifi_X = codifica_one_hot(Xc)

    print("La codificacion es: ",codifi_X)
    print("--------------------------------------------")



# =====================================================
# EJERCICIO 8: APLICACIONES DEL CLASIFICADOR MULTICLASE
# =====================================================


# ---------------------------------------------------------
# 8.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación OvR Y one-hot de los ejercicios anteriores,
# para obtener un clasificador que aconseje la concesión, 
# estudio o no concesión de un préstamo, basado en los datos X_credito, y_credito. 

# Ajustar adecuadamente los parámetros (nuevamente, no es necesario ser demasiado 
# exhaustivo)

# ----------------------


def test8_1(muestras_randoms):
    #######params###########

    #########################
    print("--------------------------- TEST 8.1 ovr + onehot-----------------")
    X = cd.X_votos
    y = cd.y_votos
    test = 0.3

    X_tr,X_te,y_tr,y_te = particion_entr_prueba(X, y, test)

    conjunto_codeado = codifica_one_hot(X_tr)


    clasif_0vr = RL_OvR(0.01,False,64)
    clasif_0vr.entrena(X_tr,y_tr,100,False)

    _, y_te_procesadas = procesar_y(np.unique(y_te),y_te)
    tasa = rendimiento(clasif_0vr,X_te,y_te)

    for i in range(0,muestras_randoms):
        ejemplo_random = np.random.randint(0,len(X)-1)
        dato_ejemplo = X[ejemplo_random]
        print(f"*Se pretende clasificar el ejemplo con el indice {ejemplo_random}: {dato_ejemplo}")

        prediccion_ejemplo = clasif_0vr.clasifica([dato_ejemplo])
        print(f"{y[ejemplo_random] == prediccion_ejemplo[0]} -> Clasificacion real: {y[ejemplo_random]}; Predicción del modelo: {prediccion_ejemplo[0]}")
    
    print("Rendimiento test:",tasa)
    print("--------------------------------------------")


# ---------------------------------------------------------
# 8.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación OvR anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 


# --------------------------------------------------------------------------
import os
import zipfile

# Funcion leer los datos de imagenes
def caragar_imagenes(ruta):
    # Abrimos el arcvhivo en modo r (lectura(read)) 
    with open(ruta, "r") as f:
        # Almacenamos en lineas todas las lineas que las lee en el archivo
        lineas = f.readlines()
    # Ahora con la variable imagenes almacenamos todas las imagenes (filas las caracteristicas) proceseadas
    imagenes = []
    for linea in lineas:
        # aqui utlizamos np.where en lugar de un bucle for , he utlizado np.logical_or para si vea un 
        # "+" (borde del dígito) o "#" (interior del dígito) le asigna '0' pixel negro sino 
        # que sea un espacio le asgina '1' pixel blanco que al final obetenmos una matriz binaria con 0s y 1s
        imagen = np.where(np.logical_or(linea.strip() == '+', linea.strip() == '#'), 0, 1)
        # lo almacenamos en imagenes
        imagenes.append(imagen)
    # convertimos las imagenes a un array de numpy
    return np.array(imagenes)

# Funcion para leer las clsificaciones de las imagenes
def cargar_etiquetas(ruta):
    # creamos la variable etiquetas para almacenar las etiquetas
    etiquetas = []
    # Abrimos el archvio con le modo r
    with open(ruta, "r") as f:
        # iteramos cada linea y lo almacenamos en etiqutas como entero elimnando los espacios blancos
        for linea in f:
            etiquetas.append(int(linea.strip()))
    # convertimos las etiquetas a un array de numpy
    return np.array(etiquetas)

def extrair_datos_zip(nom_zip):
    with zipfile.ZipFile(nom_zip, "r") as zip:
        zip.extractall("Trabajo AIA/datos/digitos")
    
    ruta = os.path.join("Trabajo AIA/datos/digitos", "")

    X_training = caragar_imagenes(os.path.join(ruta,"trainingimages"))
    X_valid = caragar_imagenes(os.path.join(ruta,"validationimages"))
    X_test = caragar_imagenes(os.path.join(ruta,"testimages"))

    y_training = cargar_etiquetas(os.path.join(ruta,"traininglabels"))
    y_valid = cargar_etiquetas(os.path.join(ruta,"validationlabels"))
    y_test = cargar_etiquetas(os.path.join(ruta,"testlabels"))

    return X_training, y_training, X_valid, y_valid, X_test, y_test

def test8_2():
    X_training, y_training, X_valid, y_valid, X_test, y_test = extrair_datos_zip("Trabajo AIA/datos/digitdata.zip")
    print("shape_Xtr :", X_training.shape)
    print("shape_ytr :", y_training.shape)
    print("shape_XV :", X_valid.shape)
    print("shape_yV :", y_valid.shape)
    print("shape_Xt :", X_test.shape)
    print("shape_yt :", y_test.shape)

#    modelo = RL_OvR(rate=0.1,rate_decay=False,batch_tam=64)
#    modelo.entrena(X_training,y_training,salida_epoch=False)

#    rend_valid = rendimiento(modelo, X_valid,y_valid)
#    rend_test = rendimiento(modelo, X_test, y_test)

#    print(f"Rendimiento en validacion: {rend_valid}")
#    print(f"Rendimiento en test: {rend_test}")


# =========================================================================
# EJERCICIO OPCIONAL PARA SUBIR NOTA: 
#    CLASIFICACIÓN MULTICLASE CON REGRESIÓN LOGÍSTICA MULTINOMIAL
# =========================================================================


#  Se pide implementar un clasificador para regresión
#  multinomial logística con softmax (VERSIÓN MINIBATCH), descrito en las 
#  diapositivas 55 a 57 del tema de "Complementos de Aprendizaje Automático". 

# class RL_Multinomial():

#     def __init__(self,rate=0.1,rate_decay=False,
#                   batch_tam=64):

#        ......

#     def entrena(self,X,y,n_epochs=100,salida_epoch=False):

#        .......

#     def clasifica_prob(self,ejemplos):

#        ......
 

#     def clasifica(self,ejemplos):

#        ......
   

 
# Los parámetros tiene el mismo significado que en el ejercicio 7 de OvR. 

# En eset caso, tiene sentido definir un clasifica_prob, ya que la función
# softmax nos va a devolver una distribución de probabilidad de pertenecia 
# a las distintas clases. 


# NOTA 1: De nuevo, es muy importante para la eficiencia usar numpy para evitar
#         el uso de bucles for convencionales.  

# NOTA 2: Se recomienda usar la función softmax de scipy.special: 

    # from scipy.special import softmax   
#

    
# --------------------------------------------------------------------

# Ejemplo:

# >>> rl_iris_m=RL_Multinomial(rate=0.001,batch_tam=8)

# >>> rl_iris_m.entrena(Xe_iris,ye_iris,n_epochs=50)

# >>> rendimiento(rl_iris_m,Xe_iris,ye_iris)
# 0.9732142857142857

# >>> rendimiento(rl_iris_m,Xp_iris,yp_iris)
# >>> 0.9736842105263158
# --------------------------------------------------------------------

# --------------- 



from scipy.special import softmax
class RL_Multinomial():

    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.clases = []
        self.weights = None
        self.biases = None

    def inicializacion_pesos(self, n_carac, n_clases):
        self.weights = np.random.uniform(low=-0.2, high=0.2, size=(n_carac, n_clases))
        self.biases = np.zeros(n_clases)

    def entropia_cruzada(self, X, y):
        z = np.dot(X, self.weights) + self.biases
        # prob partenencia a cada clase axis = 1 (filas)
        prob = softmax(z, axis=1)
        log_prob = np.log(np.maximum(prob, 1e-10))  # evitar los logs de ceros
        EC = -np.sum(log_prob * y) / X.shape[0]
        return EC

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        self.clases = np.unique(y)
        n_carac = X.shape[1]
        n_clases = len(self.clases)

        # Normalizar input data
        norm = NormalizadorStandard()
       # X = codifica_one_hot(X)
        norm.ajusta(X)
        X = norm.normaliza(X)

        # codifica y con codifica_one_hot
        codifica_y = codifica_one_hot(y)

        # Initialize weights and biases
        self.inicializacion_pesos(n_carac, n_clases)

        for epoch in range(n_epochs):
            if self.rate_decay:
                self.rate *= 1 / (1 + n_epochs)

            # Mini-batch training
            for i in range(0, X.shape[0], self.batch_tam):
                
                X_batch = X[i:i + self.batch_tam]
                y_batch = codifica_y[i:i + self.batch_tam]

                z = np.dot(X_batch, self.weights) + self.biases
                y_pred = softmax(z, axis=1)

                gradient_weights = np.dot(X_batch.T, y_pred - y_batch)
                # como queremo obtener la suma de las difrenecias para cada clase por separado 
                # axis = 0 suma a lo largo del eje de las filas
                gradient_biases = np.sum(y_pred - y_batch, axis=0)

                self.weights -= self.rate * gradient_weights
                self.biases -= self.rate * gradient_biases

            if salida_epoch:
                EC = self.entropia_cruzada(X, codifica_y)
                rend = self.rendimiento(X, y)
                print(f"Epoch {epoch + 1} - Entropia Cruzada: {EC} - Rendimiento: {rend}")

    def clasifica_prob(self, ejemplos):
        if self.weights is None or self.biases is None:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado.")

        z = np.dot(ejemplos, self.weights) + self.biases
        prob = softmax(z, axis=1)
        return prob

    def clasifica(self, ejemplos):
        prob = self.clasifica_prob(ejemplos)
        # obtenemos la clase (columna) que tiene mayor probabilidad (filas)
        pred = np.argmax(prob, axis=1)
        return self.clases[pred]

    def rendimiento(self, X, y):
        pred= self.clasifica(X)
        aciertos = np.sum(pred == y)
        return aciertos / y.shape[0]


#TEST
def test_OP():
    
    X = cd.X_votos
    y = cd.y_votos
    test = 0.3
    print("--------------------------- TEST OP  Rl-mult-----------------")
    rend_tes = 0
    rend_train = 0
    i = 0
    while((rend_tes and rend_train) < 0.55):
        # print("Intento ",i)
        X_training,X_test,y_training,y_test = particion_entr_prueba(X, y, test)
    
        modelo = RL_Multinomial(rate=0.1,batch_tam=16,rate_decay=False)
        modelo.entrena(X_training, y_training,n_epochs=100, salida_epoch=True)
        rend_tes = rendimiento(modelo, X_test, y_test)
        rend_train = rendimiento(modelo, X_training, y_training)
        i+=1

    print("Rendimiento de test: ", rend_tes)
    print("Rendimiento de train: ", rend_train)
    print("--------------------------------------------")

    
#test1()
#test2_1()
#test2_2()
#test3()
#test4()
#test5()
#test6()
#test7()
#test8_1(10)
#test8_2()
test_OP()