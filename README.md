# Segment Anything Model (SAM)

Le modèle **Segment Anything (SAM)** produit des masques d'objets de haute qualité à partir d'entrée telles que des points ou des boîtes, et il peut être utilisé pour générer des masques pour tous les objets d'une image. Il a été entraîné sur un ensemble de données de 11 millions d'images et de 1,1 milliard de masques.


# Segmentation d'images

**SAM** permet de segmenter des images en créant plusieurs masques pour détourer des objets (voir [la démo](https://segment-anything.com/demo))

## Vision Transformer (ViT)

Les ViT ont été conçus pour traiter des images de manière plus efficace et avec moins de données d'entraînement en utilisant une approche basée sur les Transformers.

Les Transformers ont été introduits pour le traitement du langage naturel et ont depuis lors été appliqués avec succès à d'autres tâches, telles que la traduction automatique et la génération de texte.

#### Fonctionnement du ViT

Le fonctionnement des ViT est assez simple. Tout d'abord, l'image est divisée en une grille de petites images carrées appelées "patches". Chaque patch est ensuite linéarisé et transmis à une couche d'attention, qui calcule les poids d'attention pour chaque patch en fonction de ses relations avec les autres patches.

Les poids d'attention sont ensuite utilisés pour agréger les informations de chaque patch en un seul vecteur de caractéristiques, qui est transmis à une série de couches de transformation linéaire pour produire une représentation de l'image.

Les ViT peuvent être utilisés pour la segmentation d'images en utilisant une approche appelée "segmentation basée sur l'attention" ou "attention-based segmentation". Dans cette approche, les ViT sont utilisés pour extraire des caractéristiques à partir de l'image, puis une couche d'attention est utilisée pour attribuer une étiquette de segment à chaque pixel de l'image en prenant en compte les relations spatiales entre les pixels.

#### Les Vision Transformer MAE

![ViT MAE](https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png)

Un Vision Transformer MAE (Merged Attention Embeddings) est une variante des Vision Transformers (ViT) qui utilise une méthode de fusion d'attention pour améliorer la performance de la classification d'images.

Le Vision Transformer MAE utilise une approche différente pour traiter les relations spatiales entre les différentes parties de l'image. Au lieu de traiter chaque patch individuellement, le MAE combine plusieurs patches en un seul vecteur d'attention à l'aide d'une méthode de fusion d'attention.

La méthode de fusion d'attention utilise les poids d'attention calculés pour chaque patch pour calculer un nouveau vecteur d'attention qui capture les relations spatiales entre les différents patches. Ce vecteur d'attention est ensuite utilisé pour agréger les informations de chaque patch en un seul vecteur de caractéristiques, qui est transmis aux couches de transformation linéaire pour la classification.

## Zero-shot learning

En apprentissage automatique, le zero-shot learning (apprentissage sans étiquette) est une approche qui permet de classifier des données sans avoir besoin de données d'entraînement pour toutes les classes. Au lieu de cela, cette méthode utilise des informations supplémentaires pour généraliser à de nouvelles classes non vues pendant l'entraînement.

Plus précisément, cette technique consiste à entraîner un modèle à prédire des propriétés ou des attributs qui décrivent les données plutôt que de prédire directement leur classe. Ces propriétés ou attributs peuvent inclure des descriptions textuelles ou des caractéristiques visuelles, et sont souvent définis en amont ou annotés manuellement. Une fois que le modèle a été entraîné sur ces propriétés ou attributs, il peut être utilisé pour classer de nouvelles données en fonction de ces propriétés ou attributs, même si ces données n'ont jamais été vues auparavant.

L'apprentissage sans étiquette est particulièrement utile lorsque les classes sont difficiles ou coûteuses à annoter, ou lorsque de nouvelles classes apparaissent régulièrement et doivent être ajoutées au modèle sans devoir re-entraîner complètement ce dernier.

## Fonctionnement de SAM

![SAM Schema](https://scontent-cdg4-1.xx.fbcdn.net/v/t39.2365-6/338558258_1349701259095991_4358060436604292355_n.png?_nc_cat=104&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=dAJexbuymEgAX8sEQ6Z&_nc_ht=scontent-cdg4-1.xx&oh=00_AfDIzJFjII7sj1dgjvpeEhSPFCPDhJiVD1zhrIAheZFMAQ&oe=644FE709)

Le Segment Anything Model se compose de 3 éléments : un image encoder, un prompt encoder et un mask decoder.

**L'image encoder :** C'est dans cette partie que se trouve le ViT MAE. Cet image encoder ne tourne qu'une seule fois par image

**Le prompt encoder :** On distingue deux types de prompt, *sparse* (points, boîtes et texte) et *dense* (masque). Les points et les boîtes sont représentés par leur coordonnées et pour chaque type de prompt on y ajoute un "learned embedding". Un learned embedding est une représentation vectorielle apprise automatiquement par un algorithme de machine learning à partir des données d'entrée. Pour le texte, on utilise un text encoder de CLIP, c'est un modèle de traitement du langage naturel et de vision par ordinateur conçu pour effectuer des tâches de correspondance de texte à image (recherche d'image, annotation automatique ou classification).

Pour les prompt *dense*, des convolutions sont utilisées, c'est une opération mathématique utilisée dans les réseaux de neurones convolutifs afin d'extraire des caractéristiques des images.

**Le mask decoder :** Le rôle du mask decoder est de lier l'image embedding, les prompt embeddings et un output token à un masque.

Le modèle Segment Anything Model utilise une architecture de Transformer similaire à celle utilisée dans les modèles de traitement de langage naturel. Cependant, au lieu de traiter des séquences de mots, le modèle Segment Anything Model traite des images divisées en une grille de patches. Cette grille de patches est transformée en une séquence de vecteurs d'embedding en utilisant une couche "patch embedding" (encodage de patch), qui permet de transformer chaque patch d'image en un vecteur de dimension fixe.

Ensuite, la séquence de vecteurs d'embedding est passée à travers plusieurs couches d'encodage de Transformers, qui permettent au modèle d'apprendre des relations spatiales à longue portée entre les patches d'image. Ces couches d'encodage utilisent des mécanismes d'attention multi-têtes pour donner plus de poids aux patches importants de l'image lors de la segmentation.

Après la phase d'encodage, le modèle Segment Anything Model utilise une couche de décodage qui utilise des convolutions pour prédire une carte de segmentation pixel par pixel. Cette couche de décodage utilise également des informations des couches d'encodage précédentes pour guider la prédiction des pixels de la carte de segmentation.

Pour éviter les ambiguïtés, le modèle ne produit pas 1 mais 3 sorties (masques) pour couvrir tous les cas ambiguës possibles. Par exemple, pour chaque masque demandé, le modèle produira : l'entièreté, une partie, une sous-partie. Pour juger la qualité du masque, le modèle calcule également un score de confiance.

L'ensemble du design du model est motivé par l'efficacité. Avec un image embedding précalculé, le prompt encoder et la mask decoder tournent dans un navigateur web en ~50ms en utilisant le CPU.

## Focal Loss et Dice Loss

Le Focal Loss et le Dice Loss sont deux fonctions de coût couramment utilisées dans la segmentation d'images.

Le Focal Loss est une fonction de coût conçue pour résoudre le problème de déséquilibre de classes, où certaines classes peuvent avoir beaucoup moins d'exemples d'entraînement que d'autres. La fonction de coût Focal Loss attribue un poids plus élevé aux exemples mal classés pour les classes minoritaires, ce qui permet de mieux tenir compte de ces classes lors de l'entraînement.

Le Dice Loss (ou Coefficient de Dice) est une fonction de coût qui mesure la similarité entre deux ensembles de valeurs binaires, généralement utilisée pour évaluer la qualité des segmentations d'images. Cette fonction de coût mesure la proportion de pixels correctement classés dans la segmentation prédite par rapport à la segmentation de référence. Le Dice Loss est plus adapté pour la segmentation d'images que pour la classification, car il prend en compte la similitude spatiale entre les régions d'objet.

La combinaison de ces deux fonctions de coût est utilisée pour superviser la prédiction des masques.

## Les limitations de SAM

Le modèle de segmentation d'images Segment Anything Model (SAM) est performant dans l'ensemble, mais il n'est pas parfait. Il peut manquer de fines structures, halluciner de petites composantes déconnectées à certains moments et ne produit pas de frontières aussi nettes que des méthodes plus intensives en termes de calculs qui "zooment", par exemple. En général, il est attendu que des méthodes de segmentation interactive dédiées surpassent SAM lorsqu'un grand nombre de points sont fournis. Contrairement à ces méthodes, SAM est conçu pour la généralité et la large utilisation plutôt que pour la segmentation interactive avec un score d'IoU élevé. De plus, SAM peut traiter des demandes en temps réel, mais sa performance globale n'est pas en temps réel lorsqu'un encodeur d'image lourd est utilisé. Bien que SAM puisse accomplir de nombreuses tâches, il est difficile de concevoir des requêtes simples qui implémentent la segmentation sémantique et panoptique. Enfin, il existe des outils spécifiques à certains domaines, qui devraient surpasser SAM dans leurs domaines respectifs.


