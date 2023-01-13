# Restricted Boltzmann Machine

Implementing a **Restricted Boltzman Machine (RBM)** from scratch. Las RBM son un tipo de red generativa que ha pesar de su simplicidad son de gran utilidad como estrategia de **reducción de dimensionalidad**. A diferencia de las Máquinas No Restringidas de Boltzmann, donde una neurona puede conectarse a neuronas presentes en su capa o cualquier otra, en las RBM las neuronas no pueden conectarse a neuronas presentes en la misma capa. 

Este tipo de red neuronal consta de **2** capas, la primera capa (capa de entrada) se compone de tantas neuronas como variables de entrada. Esta capa es también llamada **capa visible**. Asimismo, las RBMs constan de una segunda capa, también conocida como **capa oculta**, la cual es en además la capa de salida de la neurona. Por lo tanto, tenemos una **matriz de pesos (W)** que es **compartida** por ambas capas, a la vez que cada una de estas posee su propio **bias**. Entonces, la función de costo consiste en **minimizar** la diferencia entre los datos **originales** y la **reconstrucción** de estos.

One purpose of deep learning models is to encode dependencies between variables. The capturing of dependencies happens through associating of scalar energy to each configuration of the variables, which serves as a measure of compatibility. High energy means bad compatibility. An energy-based model tries always to minimize a predefined energy function. The energy function for the RBMs is defined as:

$E(v, h) = - \sum_{i \in visible} a_i v_i - \sum_{j \in hidden} b_j h_j - \sum_{ij} v_i h_j w_{ij}$,

where $v_i$, $h_j$ are the binary states of visible unit $i$ and hidden unit $j$, $a_i$, $b_j$ are their biases and $w_{ij}$ is the weight between them. As can be noticed the value of the energy function depends on the configurations of visible/input states, hidden states, weights, and biases. The training of RBM consists of the finding of parameters for given input values so that the energy reaches a minimum.

Ya que RBMs son **grafos no dirigidos**, **no** ajustan su peso a traves del **descenso del gradiente** o **backpropagation**, sino que hacen uso de un proceso llamado **divergencia contrastiva**.



