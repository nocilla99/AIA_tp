import numpy as np
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
        self.weight = np.random.uniform(low=-0.2, high=0.2,size=(n_carac,1))
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

        regularizacion = 2*np.sum(self.weight**2)

        coste += regularizacion
        return coste / len(array_entropias)
    
    
    # HE UTILIZADO CASI LO MISMO QUE EL TUYO CAMBIANDO ALGUNAS COSAS PORQUE HE AÑADIDO UNA FUNCION AUXILIAR QUE 
    # ES entropia_cruzada , rendimiento  y inicializar_pesos 
    def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False, early_stopping=False,paciencia=3):
        self.clases = list(np.unique(y))
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
        
        
        

        mejor_entropia = np.Infinity
        epochs_sin_mejora = 0
        rate = self.rate
        # INICIALIZA LOS PESOS
        self.weight , self.bias = self.inicializar_pesos(X.shape[1])
        
        #Separar en batchs        
        X_partes = [X[i:i + self.batch_tam] for i in range(0, len(X), self.batch_tam)]
        y_partes = [y[i:i + self.batch_tam] for i in range(0, len(y), self.batch_tam)]

        for epoch in range(self.n_epochs):

            if(self.rate_decay):
                rate = (rate)*(1/( 1 + self.n_epochs))

            #parte de minibatch
            for i in range(len(self.weight)):
                
                for minibatch in range(len(X_partes)):
                    suma = 0
                    suma_bias = 0
                    
                    conjunto_x = X_partes[minibatch]
                    conjunto_y = y_partes[minibatch]
                    
                    # tamaños = np.unique([len(X_partes[z]) for z in range(0,len(X_partes))])

                    #wi ← wi + η*sum j∈B( [(y(j) − σ(w*x(j)))x_i(j)])

                    for j in range(len(conjunto_x)):
                        aux = np.dot(conjunto_x[j], self.weight) + self.bias
                        calculo = (conjunto_y[j] - sigmoide(aux[0])) * conjunto_x[j][i]
                        suma += calculo
                        #el error de prediccion
                        suma_bias += sigmoide(aux[0]) - conjunto_y[j]

                    self.weight[i] += rate*suma
                    self.bias -= suma_bias

            if(early_stopping or salida_epoch):

                ec_Xv = self.entropia_cruzada(Xv, yv)
                
                if(salida_epoch):
                    
                    ec_X = self.entropia_cruzada(X, y)
                    
                    rendimiento_X = self.rendimiento(X,y)
                    
                    rendimiento_Xv = self.rendimiento(Xv,yv)

                    print(f"Epoch {epoch +1}, en entrenamiento EC: {ec_X}, rendimiento: {rendimiento_X}.")
                    print(f"         en validación    EC: {ec_Xv}, rendimiento: {rendimiento_Xv}.")

                if(early_stopping): 
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



from scipy.special import expit    

def sigmoide(x):
    return expit(x)
# ---------------Metodos y clases

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


        
class ClasificadorNoEntrenado(Exception): pass

class NormalizadorNoAjustado(Exception): pass