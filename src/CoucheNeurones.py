import numpy as np

class CoucheNeurones:
    def __init__(self, n_input, n_neurons, activation='sigmoid', learning_rate=0.01):
        """
        Initialise une couche de neurones

        Parameters:
        - n_input: nombre d'entrées
        - n_neurons: nombre de neurones dans cette couche
        - activation: fonction d'activation ('sigmoid', 'tanh', 'relu')
        - learning_rate: taux d'apprentissage
        """
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.activation_name = activation
        self.learning_rate = learning_rate

        # Initialisation Xavier/Glorot pour éviter l'explosion/disparition des gradients
        limit = np.sqrt(6 / (n_input + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_input))
        self.bias = np.zeros((n_neurons, 1))

        # Variables pour stocker les valeurs lors de la propagation
        self.last_input = None
        self.last_z = None
        self.last_activation = None

        # Import de la fonction d'activation du TP précédent
        from src.activation import ActivationFunction
        self.activation_func = ActivationFunction(activation)

    def forward(self, X):
        """
        Propagation avant
        X: matrice d'entrée (n_features, n_samples)
        """
        # TODO: Implémenter la propagation avant
        # Stocker les valeurs intermédiaires pour la rétropropagation
        self.last_input = X
        self.last_z = np.dot(self.weights, X) + self.bias  # z = W·X + b
        self.last_activation = self.activation_func.apply(self.last_z)

        return self.last_activation

    def backward(self, gradient_from_next_layer):
        """
        Rétropropagation
        gradient_from_next_layer: gradient venant de la couche suivante
        """
        # TODO: Calculer les gradients par rapport aux poids et biais
        # TODO: Calculer le gradient à propager vers la couche précédente

        # Gradient par rapport à la fonction d'activation
        grad_activation = 0

        # Gradient par rapport aux poids
        grad_weights = 0

        # Gradient par rapport aux biais  
        grad_bias = 0

        # Gradient à propager vers la couche précédente
        grad_input = 0

        # Mise à jour des paramètres
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

        return grad_input