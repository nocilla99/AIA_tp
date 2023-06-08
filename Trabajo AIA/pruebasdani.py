#aqui voy a ir desarrollando los metodos antes de ponerlos en el docu del trabajo
import numpy as np
class RegresionLogisticaMiniBatch():

    def __init__(self,rate=0.1,rate_decay=False,n_epochs=100,batch_tam=64):
        self.rate= rate
        self.rate_decay = rate_decay
        self.n_epochs = n_epochs
        self.batch_tam = batch_tam
        self.clases = list()



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

        for i in range(self.n_epochs):

            #para el descenso por gradiente hace falta rate y el gradiente
            if(self.rate_decay):
                rate= (rate)*(1/(1+n))

            #TODO parte del gradiente

            if(early_stopping or salida_epoch):
                ec_Xv = calcula_EC(Xv, yv)
                
                if(salida_epoch):
                    ec_X = calcula_EC(X, y)
                    rendimiento_X = calculaPrecision(X, y)
                    rendimiento_Xv = calculaPrecision(Xv, yv)

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
        
     
    def clasifica_prob(self,ejemplos):
        pass


    def clasifica(self,ejemplo):
        pass

    pass

def calcula_EC(X,y):
    predicciones = sigmoide(X)
    predicciones_clasificadas= np.round(predicciones)
    #media de las entropias cruzadas de cada ejemplo x
    # La formula de la ec es (-y * log(X) - (1 - y) * log(1 - X)), pero piden usar where. Habra que hacer las medias de cuando "y[i]" valga 1 y cuando sea 0
    entropia_cruzada = np.mean(np.where(y == 1, -np.log(predicciones_clasificadas), -np.log(1 - predicciones_clasificadas)))

    return entropia_cruzada

#parecida a la que se da para evaluar un clasificador
def calculaPrecision(X,y):
    predicciones = sigmoide(X)
    predicciones_clasificadas= np.round(predicciones)

    return sum(predicciones_clasificadas == y)/y.shape[0]


def sigmoide(x):
    return expit(x)



