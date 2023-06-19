# HE UTILIZADO CASI LO MISMO QUE EL TUYO CAMBIANDO ALGUNAS COSAS PORQUE HE AÑADIDO UNA FUNCION AUXILIAR QUE 
    # ES entropia_cruzada , rendimiento  y inicializar_pesos 
    # def entrena(self,X,y,Xv=None,yv=None,n_epochs=100,salida_epoch=False, early_stopping=False,paciencia=3):

    #     self.clases, y = procesar_y(np.unique(y), y)
    #     self.n_epochs = n_epochs

    #     #normalizar datos entradas
    #     norm = NormalizadorStandard()
    #     norm.ajusta(X)
    #     X = norm.normaliza(X)
        

    #     if(Xv is None or yv is None):
    #         yv = y
    #         Xv = X
    #     else :    
    #         norm.ajusta(Xv)
    #         Xv = norm.normaliza(Xv)
    #         _, yv = procesar_y(np.unique(yv), yv)
        
        

    #     mejor_entropia = np.Infinity
    #     epochs_sin_mejora = 0
    #     rate = self.rate
    #     # INICIALIZA LOS PESOS
    #     self.weight , self.bias = self.inicializar_pesos(X.shape[1])
        
    #     #Separar en batchs  
    #     partes = self.batch_tam      
    #     X_partes = np.array_split(X, (len(X)/self.batch_tam))
    #     # [X[i:i + self.batch_tam] for i in range(0, len(X), self.batch_tam)]
    #     y_partes = np.array_split(y, (len(y)/self.batch_tam))

    #     for epoch in range(self.n_epochs):

    #         if(self.rate_decay):
    #             rate = (rate)*(1/( 1 + self.n_epochs))

    #         #parte de minibatch
    #         for i in range(len(self.weight)):
                
    #             for minibatch in range(len(X_partes)):
    #                 suma = 0
    #                 suma_bias = 0
                    
    #                 conjunto_x = X_partes[minibatch]
    #                 conjunto_y = y_partes[minibatch]
                    
    #                 # tamaños = np.unique([len(X_partes[z]) for z in range(0,len(X_partes))])

    #                 #wi ← wi + η*sum j∈B( [(y(j) − σ(w*x(j)))x_i(j)])

    #                 for j in range(len(conjunto_x)):
    #                     aux = np.dot(conjunto_x[j], self.weight) + self.bias
    #                     calculo = (conjunto_y[j] - sigmoide(aux[0])) * conjunto_x[j][i]
    #                     suma += calculo
    #                     #el error de prediccion
    #                     suma_bias += sigmoide(aux[0]) - conjunto_y[j]

    #                 self.weight[i] += rate*suma
    #                 self.bias -= suma_bias

    #         if(early_stopping or salida_epoch):

    #             ec_Xv = self.entropia_cruzada(Xv, yv)
                
    #             if(salida_epoch):
                    
    #                 ec_X = self.entropia_cruzada(X, y)
                    
    #                 rendimiento_X = self.rendimiento(X,y)
                    
    #                 rendimiento_Xv = self.rendimiento(Xv,yv)

    #                 print(f"Epoch {epoch +1}, en entrenamiento EC: {ec_X}, rendimiento: {rendimiento_X}")
    #                 print(f"         en validación    EC: {ec_Xv}, rendimiento: {rendimiento_Xv}")

    #             if(early_stopping): 
    #                 if ( ec_Xv > mejor_entropia):
    #                     epochs_sin_mejora += 1

    #                     if(epochs_sin_mejora>=paciencia):
    #                         print("~~~~~~~~~~PARADA TEMPRANA~~~~~~~~~~")
    #                         break
    #                 else:
    #                     mejor_entropia = ec_Xv
    #                     epochs_sin_mejora = 0