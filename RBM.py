import numpy as np

def l2_cost(Yp, Yr):
    return np.mean((Yr - Yp) ** 2)

class RBM:
    def __init__(self, n_input_neurons, n_output_neurons):
        self.W = np.random.normal(loc=0.0, scale=1.0, size=(n_input_neurons, n_output_neurons)).astype(np.float32)
        self.visible_bias = np.random.rand(1, n_input_neurons) #visible layer bias
        self.hidden_bias = np.random.rand(1, n_output_neurons) #hidden layer bias
    
    def __sample(self, probability_distribution):
        return probability_distribution > np.random.random_sample(size=probability_distribution.shape)
        """
        random_dist = np.random.uniform(0, 1, probability_distribution.shape)
        example = probability_distribution - random_dist
        
        example[example > 0] = 1.0
        example[example <= 0] = 0.0
        return example
        """
    
    def __sigmoid(self, x):
        return 1 / (1 + np.e ** (-x))
        
    def __encode(self, X):
        probability_distribution = self.__sigmoid(X @ self.W + self.hidden_bias) #probabilities of the hidden units
        return probability_distribution, self.__sample(probability_distribution)
        
    def __decode(self, X):
        probability_distribution = self.__sigmoid(X @ self.W.T + self.visible_bias) #probabilities of the visible units
        return probability_distribution, self.__sample(probability_distribution)
    
    def getReconstructedOutput(self, X):
        encode_probability, encode_sample = self.__encode(X)
        decode_probability, decode_sample = self.__decode(encode_sample)
        return decode_probability
        
    def train(self, X, loss_function, lr=.01, epochs= 500, verbose = False):
        epoch = 0
        history = []
        while epoch < epochs:
            h0_prob, h0_state = self.__encode(X)
            positive_associations = X.T.dot(h0_prob)
            
            v1_prob, v1_state = self.__decode(h0_state)
            h1_prob, h1_state = self.__encode(v1_state)            
            negative_associations = v1_state.T.dot(h1_prob)
            
            #Updating weights
            self.W+= lr * (positive_associations - negative_associations)
            self.hidden_bias+= (lr *  (h0_prob.sum(axis = 0) - h1_prob.sum(axis = 0)) )
            self.visible_bias+= (lr * (X.sum(axis=0) - v1_state.sum(axis=0)) )
            
            epoch+=1
            loss = loss_function(v1_state, X) #loss
            history.append(loss)
            if verbose:
                print(f'Epoch {epoch} ==> Loss: {loss}')
        return history
