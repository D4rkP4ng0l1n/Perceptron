import numpy as np
from src.PerceptronMultiCouches import PerceptronMultiCouches

def test_xor():
    """
    Test du réseau multicouches sur le problème XOR
    """
    # Données XOR
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    print("Test sur le problème XOR")
    print("Données d'entrée :")
    print(X_xor)
    print("Sorties attendues :")
    print(y_xor.flatten())

    # Essayez différentes architectures
    architectures = [
        [2, 2, 1],    # 2 entrées, 2 neurones cachés, 1 sortie
        [2, 3, 1],    # 2 entrées, 3 neurones cachés, 1 sortie  
        [2, 4, 1],    # 2 entrées, 4 neurones cachés, 1 sortie
        [2, 2, 2, 1], # 2 couches cachées
    ]

    for arch in architectures:
        print(f"\n--- Architecture {arch} ---")

        # Créer et entraîner le réseau
        mlp = PerceptronMultiCouches(arch, learning_rate=0.5, activation='sigmoid')
        mlp.fit(X_xor, y_xor, epochs=1000, verbose=False)

        # Test des prédictions
        predictions = mlp.predict(X_xor)
        print("Prédictions :")
        for i in range(len(X_xor)):
            print(f"  {X_xor[i]} -> {predictions[i][0]:.4f} (attendu: {y_xor[i][0]})")

        # Calculer l'accuracy
        accuracy = mlp.compute_accuracy(y_xor, predictions)
        print(f"Accuracy finale : {accuracy:.4f}")

# Lancer le test
test_xor()