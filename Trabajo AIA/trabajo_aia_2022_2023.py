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
def particion_entr_prueba(X,y,test):
   
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
            indices_training += diccionario_indices[clase][tamanyo-1:]
            indices_test += diccionario_indices[clase][:tamanyo-1]
        
        
        indices_test.sort()
        indices_training.sort()

        return indices_training,indices_test
            

    '''
    Entrada X, y , test(proporcion)
    Separa los ejemplos X en conjunto de test y entrenamientos

    Salida: X_training,X_test,y_training,y_test
    '''

    ind_tr, ind_te = separar_clases_test_training(y, test)
    X_training,y_training = np.array([ X[i]for i in ind_tr] ), np.array([y[i] for i in ind_tr])

    X_test,y_test = np.array([X[i] for i in ind_te]) , np.array([y[i] for i in ind_te])
    return X_training,X_test,y_training,y_test



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
        self.weight = np.random.randn(n_carac,1)
        self.bias = np.random.randn()
        return self.weight, self.bias

    def entropia_cruzada(self, X,y,reg_lambda):
      
        y_pred = self.clasifica_prob(X)
        # selecciona los elementos correspondientes a la clase positiva y calcula el logaritmo de las probabilidades de prediccion 
        # para esos elementos, y también calcula el logaritmo de las probabilidades complementarias 
        # para los elementos correspondientes a la clase negativa.
        coste = np.sum(np.where(y == 1, -np.log(y_pred), -np.log(1 - y_pred)))
        regularizacion = (reg_lambda / 2)*np.sum(self.weight**2)
        coste += regularizacion
        return coste 
    
    # HE HECHO ESTA FUNCION PORQUE LA FUNCION QUE ESTA EN EL ANUNCIADO ME DEVUELVE UN RENDIMIENTO CON UNA LISTA DE NUMEROS 
    # EN VEZ DE UN NUMERO SOLO AL EJECUTAR EL EJERICCIO 4
    def rendimiento(self, X, y):
        predicciones = self.clasifica(X)
        aciertos = (predicciones == y).sum()
        precision = aciertos / y.shape[0]
        return precision
    
    # HE UTILIZADO CASI LO MISMO QUE EL TUYO CAMBIANDO ALGUNAS COSAS PORQUE HE AÑADIDO UNA FUNCION AUXILIAR QUE 
    # ES entropia_cruzada , rendimiento  y inicializar_pesos 
    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False, early_stopping=False,paciencia=3):
       self.clases = list(np.unique(y))

       if(early_stopping):
            if(yv == None):
                yv = y
            if(Xv == None):
                Xv = X

       mejor_entropia = np.Infinity
       epochs_sin_mejora = 0
       rate = self.rate
       # INICIALIZA LOS PESOS
       self.weight , self.bias = self.inicializar_pesos(X.shape[1])
       
       for epoch in range(n_epochs):

            #para el descenso por gradiente hace falta rate y el gradiente
            if(self.rate_decay):
                rate = (rate)*(1/( 1 + n_epochs))

            altera = np.random.permutation(X.shape[0])
            X = X[altera]
            y = y[altera]

            for i in range(0, X.shape[0], self.batch_tam):
                conjunto_x = X[i:i + self.batch_tam]
                conjunto_y = y[i:i + self.batch_tam].reshape(-1,1)

                y_pred = self.clasifica_prob(conjunto_x)
                # he utlizado la traspuesta de la matriz poque al final del bucle tenemos conjunto_x de dimension (batch_tam, num_carac)
                # y (conjunto_y - y_pred) es deminsion (batch_tam,1) entonces al aplicar la traspuesta seria la dimension
                # de conjunto_x (num_carac , batch_tam) y asi podemos hacer la multiplicacion
                self.weight -= rate * np.dot(conjunto_x.T , (conjunto_y - y_pred))
                self.bias -= rate * np.sum(conjunto_y - y_pred)

            #TODO parte del gradiente

            if(early_stopping or salida_epoch):
                ec_Xv = self.entropia_cruzada(Xv, yv,0.1)
                
                if(salida_epoch):
                    ec_X = self.entropia_cruzada(Xv, yv,0.1)
                    rendimiento_X = self.rendimiento(X, y)
                    rendimiento_Xv = self.rendimiento(Xv, yv)
                    print(f"Epoch {epoch}, en entrenamiento EC: {ec_X}, rendimiento: {rendimiento_X}.")
                    print(f"         en validación    EC: {ec_Xv}, rendimiento: {rendimiento_Xv}.")
                else : 
                    if ( ec_Xv > mejor_entropia):
                        epochs_sin_mejora += 1

                        if(epochs_sin_mejora>=paciencia):
                            print("PARADA TEMPRANA")
                            break
                    else:
                        mejor_entropia = ec_Xv
                        epochs_sin_mejora = 0

    def clasifica_prob(self,ejemplos):
        if self.weight is None or self.bias is None:
            raise ClasificadorNoEntrenado("El clasificador no ha sido entrenado.")
        # vector z que contiene la uncion lineal para cada ejemplo despues de multiplicarle con su peso y le suma el sesgo
   
        self.weight = self.weight.astype(np.float64)
      
        z = np.dot(ejemplos, self.weight) + self.bias
        # al aplicar el sigmoide al vector z nos sale las probalidades de prediccion de cada ejemplo X
        y_pred = sigmoide(z)
        return y_pred.reshape(-1,1)


    def clasifica(self,ejemplo):
        probabilidad = self.clasifica_prob(ejemplo)
        # entonces ahora despues de obtener la probadlidad asignamos que si la prob >= 0.5 enonces su clasificacion 1
        # sino le asiganmos una clasificacion 0
        return np.where(probabilidad >= 0.5, self.clases[1], self.clases[0]).flatten()



# PARTE DANI

'''''
    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False, early_stopping=False,paciencia=3):
        self.clases = list(np.unique(y))
        self.n_epochs = n_epochs

        if(early_stopping):
            if(yv == None):
                yv = y
            if(Xv == None):
                Xv = X


        mejor_entropia = np.Infinity
        epochs_sin_mejora = 0
        rate = self.rate
        pesos = np.ones(X.shape[1])

        for i in range(self.n_epochs):

            #para el descenso por gradiente hace falta rate y el gradiente
            if(self.rate_decay):
                rate= (rate)*(1/(1+self.n_epochs))

            #TODO parte del gradiente

            if(early_stopping or salida_epoch):
                ec_Xv = calcula_EC(Xv, yv, pesos)
                
                if(salida_epoch):
                    ec_X = calcula_EC(X, y, pesos)
                    rendimiento_X = calculaPrecision(X, y, pesos)
                    rendimiento_Xv = calculaPrecision(Xv, yv, pesos)

                    print("EPOCH {}, en entrenamiento EC: {}, rendimiento: {}. \n \ten validación    EC: {},  rendimiento: {}."
                    .format(np.round(ec_X,5),np.round(rendimiento_X,5),np.round(ec_Xv,5),np.round(rendimiento_Xv,5)))
                
                else : 
                    if ( ec_Xv > mejor_entropia):
                        epochs_sin_mejora += 1

                        if(epochs_sin_mejora>=paciencia):
                            break
                    else:
                        mejor_entropia = ec_Xv
                        epochs_sin_mejora = 0
 '''   


''''' 
def calcula_EC(X,y,pesos):
    predicciones = sigmoide(np.dot(X,pesos))
    predicciones_clasificadas= np.round(predicciones)
    #suma de las entropias cruzadas de cada ejemplo x
    # La formula de la ec es (-y * log(X) - (1 - y) * log(1 - X)), pero piden usar where. Habra que hacer las medias de cuando "y[i]" valga 1 y cuando sea 0
    entropia_cruzada = np.sum(np.where(y == 1, -np.log(predicciones_clasificadas), -np.log(1 - predicciones_clasificadas)))

    return entropia_cruzada

#parecida a la que se da para evaluar un clasificador
def calculaPrecision(X,y, pesos):
    predicciones = sigmoide(np.dot(X,pesos))
    predicciones_clasificadas= np.round(predicciones)

    return sum(predicciones_clasificadas == y)/y.shape[0]
'''


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
    particiones = np.array_split(np.random.permutation(range(len(X))), n)
    # para almacenar el rendimiento de cada iteracion en la valid_cruz
    rends = []

    for i in range(n):
        # aprovechando del ej1 dividimos el conjunto de datos en tranining y validacion aleatoriamente
        #X_training,Xv,y_training,yv = particion_entr_prueba(X,y, test=1/n)
        
        indices_entrenamiento = np.concatenate([particiones[j] for j in range(n) if j != i])
        indices_validacion = particiones[i]
        
        X_training = X[indices_entrenamiento]
        y_training = y[indices_entrenamiento]
        X_validacion = X[indices_validacion]
        y_validacion = y[indices_validacion]

        # entrenamos los datos de training
        clasificador = clase_clasificador(**params)
        clasificador.entrena(X_training,y_training)

        # evaluamos el rendimineto de los daots de validacion
        rend = rendimiento(clasificador,X_validacion, y_validacion)
        rends.append(rend)
        print(f"Partición {i+1}. Rendimiento: {rend}")

    # rendimiento medio entre todos los rendimintos
    rend_med = np.mean(rends)
    return rend_med


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





#testear







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

#descargar y aplicar funciones




















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




import numpy as np
from scipy.special import softmax

class RL_Multinomial():
    def __init__(self, rate=0.1, rate_decay=False, batch_tam=64):
        self.rate = rate
        self.rate_decay = rate_decay
        self.batch_tam = batch_tam
        self.weights = None

    def entrena(self, X, y, n_epochs=100, salida_epoch=False):
        # Inicilizamos los pesos
        n_clases = len(np.unique(y))
        n_carac = X.shape[1]
        # una mtriz de pesos para multiples clases
        self.weights = np.random.randn(n_clases, n_carac)

        # 
        for epoch in range(n_epochs):
            if self.rate_decay:
                rate = (self.rate)*(1/( 1 + n_epochs))
            else:
                rate = self.rate

            altera = np.random.permutation(X.shape[0])
            X_altera = X[altera]
            y_altera = y[altera]

            for i in range(0, X.shape[0], self.batch_tam):
                conjunto_X = X_altera[i:i+self.batch_tam]
                conjunto_y = y_altera[i:i+self.batch_tam]
                gradients = self.calcular_gradientes(conjunto_X, conjunto_y)
                self.weights -= rate * gradients

            if salida_epoch:
                loss = self.calcular_entropia_cruzada(X, y,0)
                print(f"Epoch {epoch+1}/{n_epochs} - Loss: {loss}")


    # 
    def calcular_entropia_cruzada(self, X, y,reg_lambda):
        m = codifica_one_hot(y)  # Codificación one-hot de las etiquetas verdaderas
        y_pred = self.clasifica_prob(X)  # Probabilidades predichas utilizando softmax
        coste = -np.sum(np.dot(m.T , np.log(y_pred)))  # Suma de los términos de la entropía cruzada
        regularizacion = (reg_lambda)*np.sum(self.weights)
        coste += regularizacion
        return coste  

    def clasifica_prob(self, ejemplos):

        z = np.dot(ejemplos, self.weights.T)
        y_pred = softmax(z)
        return y_pred

    def clasifica(self, ejemplos):
        prob = self.clasifica_prob(ejemplos)
        return np.argmax(prob, axis=1)

