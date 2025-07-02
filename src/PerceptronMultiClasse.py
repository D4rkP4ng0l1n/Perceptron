import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PerceptronSimple import *

class PerceptronMultiClasse:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.perceptrons = {}
        self.classes = None

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne un perceptron par classe (stratégie un-contre-tous)
        """
        self.classes = np.unique(y)

        for classe in tqdm(self.classes, desc="Entraînement des perceptrons"):
            # TODO: Créer un problème binaire pour cette classe
            y_pred = np.where(classe < 0, 0, 1)
            
            # Transformer y en problème binaire : classe courante vs toutes les autres
            y_binary = np.where(y == classe, 1, -1)

            # TODO: Entraîner un perceptron pour ce problème binaire
            perceptron = PerceptronSimple(learning_rate=self.learning_rate)
            perceptron.fit(X, y_binary, max_epochs)

            # Stocker le perceptron entraîné
            self.perceptrons[classe] = perceptron

    def predict(self, X):
        """Prédit en utilisant le vote des perceptrons"""
        if not self.perceptrons:
            raise ValueError("Le modèle n'a pas été entraîné. Appelez fit() d'abord.")

        # TODO: Calculer les scores de tous les perceptrons
        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, classe in enumerate(self.classes):
            # Calculer la sommation pondérée (avant fonction d'activation)
            # pour obtenir un score de confiance
            perceptron = self.perceptrons[classe]
            raw_scores = X.dot(perceptron.weights) + perceptron.bias
            scores[:, i] = raw_scores

        # TODO: Retourner la classe avec le score maximum
        predicted_indices = np.argmax(scores, axis=1)
        return self.classes[predicted_indices]

    def predict_proba(self, X):
        """Retourne les scores de confiance pour chaque classe"""
        if not self.perceptrons:
            raise ValueError("Le modèle n'a pas été entraîné.")

        scores = np.zeros((X.shape[0], len(self.classes)))

        for i, classe in enumerate(self.classes):
            perceptron = self.perceptrons[classe]
            raw_scores = X.dot(perceptron.weights) + perceptron.bias
            scores[:, i] = raw_scores

        return scores
    
    
def evaluer_perceptron_multiclasse(X, y, target_names=None, test_size=0.3, val_size=0.5):
    """
    Évalue le perceptron multi-classes avec une méthodologie rigoureuse
    """

    # Première division : train+val / test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Deuxième division : train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp
    )

    print(f"Répartition des données :")
    print(f"  - Entraînement : {X_train.shape[0]} échantillons")
    print(f"  - Validation   : {X_val.shape[0]} échantillons")
    print(f"  - Test         : {X_test.shape[0]} échantillons")

    # Entraînement du perceptron multi-classes avec validation
    perceptron_mc = PerceptronMultiClasse(learning_rate=0.1)
    perceptron_mc.fit(X_train, y_train)

    y_train_pred = perceptron_mc.predict(X_train)
    y_val_pred = perceptron_mc.predict(X_val)
    y_test_pred = perceptron_mc.predict(X_test)

    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_val = accuracy_score(y_val, y_val_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    print(f"\nPerformances :")
    print(f"  - Accuracy train      : {accuracy_train:.3f}")
    print(f"  - Accuracy validation : {accuracy_val:.3f}")
    print(f"  - Accuracy test       : {accuracy_test:.3f}\n")

    # Affichage classification report (test)
    print("Classification report (test) :")
    print(classification_report(y_test, y_test_pred, target_names=target_names))

    # Fonction interne pour afficher matrice de confusion
    def plot_confusion(y_true, y_pred, title="Matrice de confusion"):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=target_names if target_names is not None else None,
                    yticklabels=target_names if target_names is not None else None)
        plt.ylabel('Vrai')
        plt.xlabel('Prédit')
        plt.title(title)
        plt.show()

    # Affichage matrices de confusion
    plot_confusion(y_train, y_train_pred, "Matrice de confusion - Entraînement")
    plot_confusion(y_val, y_val_pred, "Matrice de confusion - Validation")
    plot_confusion(y_test, y_test_pred, "Matrice de confusion - Test")
    
# Charger le dataset Iris
data = load_iris()
X = data.data
y = data.target
target_names = data.target_names

# Appeler la fonction d'évaluation
evaluer_perceptron_multiclasse(X, y, target_names=target_names)