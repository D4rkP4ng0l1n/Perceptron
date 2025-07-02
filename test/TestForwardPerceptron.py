import numpy as np
import unittest

from src.CoucheNeurones import CoucheNeurones
from src.PerceptronMultiCouches import PerceptronMultiCouches

class TestForwardPerceptron(unittest.TestCase):
    '''
    Fonction de test générée par IA
    '''
    def test_forward_simple(self):
        # Configuration d’un perceptron 2-3-1 avec sigmoid
        np.random.seed(0)  # Pour des résultats reproductibles
        model = PerceptronMultiCouches(architecture=[2, 3, 1], activation='sigmoid')

        # Entrée simple : 2 échantillons avec 2 features
        X = np.array([[0.0, 1.0],
                      [1.0, 0.0]])

        # Propagation avant
        y_pred = model.forward(X)

        # Vérification de la forme de sortie
        self.assertEqual(y_pred.shape, (2, 1))  # 2 échantillons, 1 sortie

        # Test de valeurs approximatives (pas de valeur exacte à tester ici à cause des poids aléatoires)
        self.assertTrue(np.all(y_pred >= 0) and np.all(y_pred <= 1))  # sigmoid ∈ [0,1]

if __name__ == '__main__':
    unittest.main()
