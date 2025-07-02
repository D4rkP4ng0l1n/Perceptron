import numpy as np
from src.CoucheNeurones import CoucheNeurones


def test_forward_sigmoid():
    '''
    Fonction de test générée par IA
    '''
    np.random.seed(42)
    couche = CoucheNeurones(n_input=2, n_neurons=1, activation='sigmoid')
    
    # Forcer des poids/biais connus pour test contrôlé
    couche.weights = np.array([[0.5, -0.5]])
    couche.bias = np.array([[0.0]])

    X = np.array([[1], [2]])  # 2 features, 1 sample
    output = couche.forward(X)
    
    # z = 0.5*1 + (-0.5)*2 = 0.5 - 1 = -0.5
    # sigmoid(-0.5) ≈ 0.3775
    expected = 1 / (1 + np.exp(0.5))   
    
    assert np.allclose(output, expected, atol=1e-4), f"Expected {expected}, got {output}"


test_forward_sigmoid()