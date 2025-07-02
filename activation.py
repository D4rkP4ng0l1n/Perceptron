import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha  # Pour Leaky ReLU

    def apply(self, z):
        if self.name == "heaviside":
            return np.where(z < 0, 0, 1)
        elif self.name == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.name == "tanh":
            return (2 / (1 + np.exp(-2 * z)) - 1)
        elif self.name == "relu":
            return np.where(z < 0, 0, z)
        elif self.name == "leaky_relu":
            return np.where(z < 0, self.alpha * z, z)
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def derivative(self, z):
        if self.name == "heaviside":
            # La dérivée de Heaviside est la distribution de Dirac
            return  np.where(z != 0, 0, 999999999999999)
        elif self.name == "sigmoid":
            return self.apply(z) * (1 - self.apply(z))
        elif self.name == "tanh":
            return  1 - self.apply(z)**2
        elif self.name == "relu":
            return np.where(z < 0, 0, 1)
        elif self.name == "leaky_relu":
            return np.where(z < 0, self.alpha, 1)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.")
            
            
z = np.linspace(-10, 10, 100)

for name in ['heaviside', 'sigmoid', 'tanh', 'relu', 'leaky_relu']:
    act = ActivationFunction(name)
    g = act.apply(z)
    d = act.derivative(z)

    plt.figure()
    plt.plot(z, g, label="Fonction d'activation", color="red")
    plt.plot(z, d, label="Dérivée", color="blue")
    plt.title("Fonction " + name + " et sa dérivée")
    plt.legend()
    plt.savefig('figures/' + name + '.png')