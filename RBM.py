import numpy as np

def l2_cost(Yp, Yr):
    return np.mean((Yr - Yp) ** 2)

class RBM:
    def __init__(self, n_input_neurons, n_output_neurons):
        #W define la 'probabilidad de importancia' que tienen las variables de entrada con respecto a las de 
        #salida y viceversa
        self.W = np.random.normal(loc=0.0, scale=1.0, size=(n_input_neurons, n_output_neurons)).astype(np.float32)
        self.visible_bias = np.random.rand(1, n_input_neurons) #visible layer bias
        self.hidden_bias = np.random.rand(1, n_output_neurons) #hidden layer bias
    
    def __sample(self, probability_distribution):
        #Hacemos 1 las probabilidades que superen a su correspondiente en una distribucion de la misma forma, por
        #ejemplo una distribucion uniforme. De esta forma decidimos cuales neuronas 'encender' o 'apagar',
        #ademas de que resulta muy util para prevenir el sobreajuste
        return probability_distribution > np.random.uniform(size=probability_distribution.shape)
    
    def __sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))
        
    def __encode(self, X):
        #Calculamos la distribucion de probabilidad de las caracteristicas en las unidades ocultas
        probability_distribution = self.__sigmoid(X @ self.W + self.hidden_bias) 
        #Identificamos las caracteristicas de mayor interes resultantes del proceso de encoding o 
        #reduccion de dimensionalidad
        return probability_distribution, self.__sample(probability_distribution)
        
    def __decode(self, X):
        #Calculamos la distribucion de probabilidad de las caracteristicas en las unidades visibles,
        #podemos interpretarlo como la reconstruccion de la imagen a partir de las caracteristicas
        #mas importantes obtenidas del encoding
        probability_distribution = self.__sigmoid(X @ self.W.T + self.visible_bias)
        #Identificamos las caracteristicas de salida mas interesantes resultantes del proceso de decoding o 
        #restauracion
        return probability_distribution, self.__sample(probability_distribution)
    
    def getReconstructedOutput(self, X):
        """
        La reconstruccion consiste en encoding los datos originales, obtener de estos las caracteristicas de
        mayor interes, aplicar estas caracteristicas sobre la matriz de pesos, y nos quedaremos con los datos
        de esta reconstruccion
        """
        encode_probability, encode_sample = self.__encode(X)
        #There is developers that decode using encode_probability, but I think that encode_sample is better
        decode_probability, decode_sample = self.__decode(encode_sample)
        return decode_probability
        
    def train(self, X, loss_function, lr=.01, epochs= 500, verbose = False):
        epoch = 0
        history = []
        while epoch < epochs:
            # Contrastive Divergence
            h0_prob, h0_state = self.__encode(X)
            v1_prob, v1_state = self.__decode(h0_state)
            h1_prob, h1_state = self.__encode(v1_state)
            
            ## Updating weights
            #dEnergy/dW = v_i * h_i,
            #so, dW = dEnergy_original/dW - dEnergy_reconstruction/dW
            delta_W = X.T.dot(h0_prob) - v1_state.T.dot(h1_prob)
            self.W+= (lr * delta_W)
            self.hidden_bias+= lr *  (h0_prob.sum(axis = 0) - h1_prob.sum(axis = 0)) 
            self.visible_bias+= lr * (X.sum(axis=0) - v1_state.sum(axis=0))
            
            epoch+=1
            #usualmente se usa como metrica de perdida el cuadrado de la diferencia entre el
            #original y la reconstruccion
            loss = loss_function(v1_state, X) #loss
            history.append(loss)
            if verbose:
                print(f'Epoch {epoch} ==> Loss: {loss}')
        return history
