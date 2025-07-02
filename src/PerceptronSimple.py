import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.error = []

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne le perceptron
        X: matrice des entrées (n_samples, n_features)
        y: vecteur des sorties désirées (n_samples,)
        """
        # Initialisation les poids et le biais
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0.0
        
        for e in tqdm(range(max_epochs)):
            nbError = 0
            for i in range(X.shape[0]):
                x = X[i]
                y_true = y[i]

                # Application de la fonction d'activation
                y_pred = np.where((self.weights*x + self.bias).sum() < 0, 0, 1)
                
                # Si la prédiction est incorrecte alors on effecue une mise à jour
                isError = y_true - y_pred
                self.weights = self.weights + self.learning_rate * isError * x
                self.bias = self.bias + self.learning_rate * isError
                
                nbError += abs(isError)
            self.error.append(nbError)        

    def predict(self, X):
        """Prédit les sorties pour les entrées X"""
        y_pred = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            x = X[i] # (n_features,)
            y_pred[i] = np.where(np.dot(self.weights, x + self.bias) < 0, 0, 1)

        return y_pred

    def score(self, X, y):
        """Calcule l'accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
        
    
def generer_donnees_separables(n_points=100, noise=2):
    """
    Génère deux classes de points linéairement séparables
    """
    np.random.seed(42)
    
    # Classe +1 autour de (2, 2)
    X_pos = np.random.randn(n_points, 2) * noise + np.array([2, 2])
    y_pos = np.ones(n_points)

    # Classe 0 autour de (-2, -2)
    X_neg = np.random.randn(n_points, 2) * noise + np.array([-2, -2])
    y_neg = np.zeros(n_points)

    # Concaténer les données
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((y_pos, y_neg))

    return X, y   

def visualiser_donnees(X, y, w=None, b=None, title="Données"):
    """
    Visualise les données et optionnellement la droite de séparation
    """
    plt.figure(figsize=(8, 6))
    # Afficher les points
    mask_pos = (y == 1)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c='blue', marker='+', s=100, label='Classe +1')
    plt.scatter(X[~mask_pos, 0], X[~mask_pos, 1], c='red', marker='*', s=100, label='Classe 0')
    # Afficher la droite de séparation si fournie
    if w is not None and b is not None:
        # TODO: Tracer la droite w·x + b = 0
        x_vals = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 100)
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
    
def analyser_convergence(X, y, learning_rates=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0]):
    """
    Analyse la convergence pour différents taux d'apprentissage
    """
    plt.figure(figsize=(12, 8))
    for i, lr in enumerate(learning_rates):
        # TODO: Entraîner le perceptron avec ce taux d'apprentissage
        perceptron = PerceptronSimple(lr)
        perceptron.fit(X, y)
        # TODO: Enregistrer l'évolution de l'erreur à chaque époque
        error = perceptron.error
        # TODO: Tracer les courbes de convergence
        plt.plot(error, label=f"Taux d'apprentissage {lr}")
        
        plt.xlabel('Époque')
        plt.ylabel("Nombre d'erreurs")
        plt.title("Convergence pour différents taux d'apprentissage")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
'''
p = PerceptronSimple()


# Données pour la fonction AND
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Données pour la fonction OR
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

p.fit(X_and, y_and)
print("Score : ", p.score(X_and, y_and))
visualiser_donnees(X_and, y_and, p.weights, p.bias)

p.fit(X_or, y_or)
print("Score : ", p.score(X_or, y_or))
visualiser_donnees(X_or, y_or, p.weights, p.bias)

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])
p.fit(X_or, y_or)
print("\nScore : ", p.score(X_or, y_or))
visualiser_donnees(X_xor, y_xor, p.weights, p.bias)


X, y = generer_donnees_separables()

p.fit(X, y)
visualiser_donnees(X, y, p.weights, p.bias)


analyser_convergence(X, y)
'''