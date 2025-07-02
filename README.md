TP1 - Perceptron

Exercice 2 :

1. Pourquoi la fonction de Heaviside pose-t-elle problème pour l'apprentissage par gradient ?
   Parce qu'elle n’est pas différentiable partout (elle est constante presque partout et a une discontinuité en 0).

2. Dans quels cas utiliser sigmoid vs tanh ?

- Sigmoid : La sortie se trouve entre 0 et 1. Donc on l'utilisera alors pour répondre à des problèmes binaire ( Si le résultat est proche de 0 alors ce sera un non et si le résultat est proche de 1 alors ce sera un oui)
- Tanh : La sortie se trouvant entre -1 et 1, la fonction est centrée sur 0. On s'en servira alors pour résoudre des problèmes plus complexes car l'apparition de valeur négative permet un apprentissage plus efficace.

3. Pourquoi ReLU est-elle si populaire dans les réseaux profonds ?
   ReLU est une fonction simple et rapide à calculer. De plus, son domaine de sortie est compris entre 0 et l'infini, ce qui fait qu'il n'y a aucune saturation pour les valeurs positives.

4. Quel est l'avantage du Leaky ReLU ?
   Leaky ReLU va corriger une erreur de ReLU. En effet, en utilisant ReLU avec une entrée négative, la fonction retournera 0 ce qui fait qu'il n'y a plus d'apprentissage. Problème qui n'existe pas avec Leaky ReLU.

Exercice 3 :

1. Que se passe-t-il si n est trop grand ?
   Si n est trop grand, le Perceptron apprend "trop vite" et il peut ne pas converger vers le bon résultat

2. Et s’il est trop petit ?
   Si n est trop petit par contre, le Perceptron cette fois apprend "trop lentement" et donc il va mettre beaucoup plus de temps à converger vers le bon résultat

3. Existe-t-il une valeur idéale de n ?
   Il n'existe pas de valeur X idéale. Il faut faire des essais avec différent n pour avoir une approximation de la valeur idéale.

4. Peut-on faire varier n au cours du temps ?
   Bien sûr.

5. Quelle stratégie pouvez vous imaginer ?
   De ce que j'ai vu sur Internet, c'est une stratégie qui est utilisé. Elle consiste à démarrer avec un n grand pour aller vite vers une bonne zone et ensuite de réduire le taux d'apprentissage pour terminer l'entraînement en étant plus précis dans la zone qu'on aura trouvé au début. Cette méthode se nomme learning rate scheduler.

Exercice 6 :

1. Quelles sont vos constatations ?
   Je constate que mon Perceptron ne peut pas créer une droite qui sépare les 2 classes comme il le faudrait

2. Quel lien peut-on faire avec la notion de séparabilité linéaire évoquée plus tôt dans le cours ?
   Le Perceptron ne peut converger que si les données sont linéairement séparables (= séparable par une droite)

Exercice 8 :

1. Quel comportement observez-vous lorsque n est très petit ?
   Lorsque n est très petit le Perceptron met beaucoup d'époque pour converger vers une solution

2. Que se passe-t-il lorsque n est trop grand ?
   Lorsque n est très grand il y a beaucoup d'oscilations et je ne peux pas dire si le Perceptron a converger vers la bonne solution

3. Existe-t-il un n optimal dans votre cas ?
   Oui, dans mon cas un n optimal serait 0.1 ou 0.01 car on voit que la courbe fini par se stabiliser même si il reste encore quelques oscillations

4. Comment la structure des données (dispersion, bruit…) peut-elle interagir avec n ?
   Plus le bruit est élevé, plus les points seront dispersé et donc plus il sera difficile pour le perceptron de séparer les classes. De ce fait, un n plus petit sera plus utile tandis que si le bruit est très faible alors les classes seront bien identifiables et donc un n grand trouvera la bonne réponse très rapidement

Exercice 9 :

1. Cohérence des prédictions : Que se passe-t-il si plusieurs perceptrons prédisent positivement pour le même exemple ?
   Si plusieurs Perceptrons prédisent positivement, peu importe les résultats, je ne garde que celui qui a le score maximal

2. Gestion des ambiguïtés : Comment gérer le cas où aucun perceptron ne prédit positivement ?
   Si il n'y a aucune prédiction positive, alors je garderais le "moins négatif" ( c'est à dire celui qui a le score le plus proche de 0 ). De toute façon ma fonction retourne toujours la classe avec le score maximum.

3. Équilibrage : Comment l'approche "Un contre Tous" gère-t-elle le déséquilibre naturel qu'elle crée ?
   L'approche "Un contre Tous" crée un déséquilibre car chaque perceptron est entraîné avec peu d'exemples positifs et beaucoup de négatifs, ce qui peut biaiser l'apprentissage à prédire "non", sauf si l'on compense ce déséquilibre par du suréchantillonnage, du sous-échantillonnage ou des pondérations.

Concernant l'exercice 10, il ne me semble pas avoir vu de consignes, je n'ai pas compris ce que je devais réaliser pour cet exercice.
Concernant l'exercice 11, honnêtement je n'ai pas compris non plus. J'ai essayé de générer un code avec Chat GPT mais je ne comprends pas comment ça fonctionne ni à quoi il sert et malgré l'aide de l'IA je n'arrive pas à comprendre.

TP2 - Multi-layer Perceptron
