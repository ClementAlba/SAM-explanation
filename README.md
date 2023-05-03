# Segment Anything Model (SAM)

Le modèle **Segment Anything (SAM)** produit des masques d'objets de haute qualité à partir d'entrée telles que des points ou des boîtes, et il peut être utilisé pour générer des masques pour tous les objets d'une image. Il a été entraîné sur un ensemble de données de 11 millions d'images et de 1,1 milliard de masques.


# Segmentation d'images

**SAM** permet de segmenter des images en créant plusieurs masques pour détourer des objets (voir [la démo](https://segment-anything.com/demo))

## Fonctionnement d'un Transformer

![Fonctionnement du Transformer](https://production-media.paperswithcode.com/methods/new_ModalNet-21.jpg)

Pour notre exemple, les inputs seront des mots d'une phrase et le Transformer doit produire en sortie une réponse à cette phrase.

### Attention

Avant tout, il est important de comprendre à quoi sert l'attention. L'attention permet de définir sur quelle partie de la phrase d'entrée nous devons nous concentrer. L'attention nous permet de mesurer et de quantifier comment les mots sont reliés entre eux dans une phrase.

### Input Embedding

La couche Input Embedding permet de prendre chaque mot en entrée et de les transformer en vecteurs contenant des valeurs décimales représentant ce mot.

En effet, un ordinateur ne comprend pas les mots, on le traduit donc par un point de l'espace défini par des coordonnées et tous les mots d'une même famille seront plus ou moins proches sur un plan.

### Positional Encoding

Il faut ajouter à chaque vecteur un vecteur de position (appelé positional embedding) permettant au Transformer de connaître la position de chaque mot, car la position des mots dans une phrase est importante. Pour créer ces vecteurs, on utilise les fonctions sinus et cosinus.

![PE formules](https://i.stack.imgur.com/PxeeE.png)

![détails PE](https://i.stack.imgur.com/mGSYD.png)

Pour les inputs pairs, on utilise la fonction sinus et pour les inputs impairs on utilise la fonction cosinus.

Ces vecteurs de position sont ensuite ajoutés à leur vecteurs d'embedding (les vecteurs représentant les mots) correspondant pour donner de nouveaux vecteurs (Positional Input Embeddings)

### Encoder Layer

Le but de l'Encoder est d'encoder l'input dans une représentation continue avec des informations d'attention. C'est ce qui va aider le Decoder à se concentrer sur les mots les plus importants durant le décodage.

Il est possible de mettre en place plusieurs Encoder et chaque couche aura la possibilité d'apprendre différentes représentations de l'attention pour booster les capacités prédictives du Transformer.

#### Multi-Headed Attention

![Attention mécanisme](https://data-science-blog.com/wp-content/uploads/2022/01/mha_img_original.png)

Les vecteurs Q (queries) et K (keys) sont multipliées entre eux pour donner une matrice de scores. Cette matrice de scores sert à déterminer à quel degré les mots dépendent des autres dans la phrase d'input. 

Ensuite, cette matrice de score est "scalé" en la divisant par la racine carré de la dimension des matrices Q et K.

On applique ensuite la fonction de softmax sur cette matrice de scores scalé, ce qui nous donne maintenant une matrice de probabilités. Grâce à l'opération de softmax, les scores les plus élevés sont augmentés, et les scores les plus faibles sont diminués.

![Softmax formule](data:image/png;base64,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)

On multiplie ensuite cette matrice par le vecteur V, ce qui nous donne un vecteur d'output.

Pour respecter le mécanisme "multi-têtes", les vecteurs Q, K et V doivent être divisés en N vecteurs avant d'y appliquer le mécanisme d'attention. Chaque processus d'attention est appelé "head", chaque "head" produit un vecteur de sortie qui est ensuite concaténé avec tous les autres vecteurs de sortie. Chaque head est censé apprendre quelque chose de différent, ce qui rend la couche Encoder encore plus puissante.

#### Normalization Layer

La couche de normalisation a pour but de mettre les valeurs à la même échelle (moyenne = 0 et variance = 1). Ce qui va permettre de réduire la dépendance des gradients entre chaque couche du réseau de neurones. Les autres gradients n'auront pas à ajuster les valeurs puisqu'elles seront toujours à la même échelle. Cela réduit donc le nombre d'étapes pour converger vers le minimum de la fonction coût et donc accélère l'apprentissage. 

#### Feed-Forward

Cette couche contient simplement un réseau de neurones "feed forward" qui est appliqué à chacun des vecteurs d'attention. Ces réseaux sont utilisés pour transformer le vecteur d'attention en quelque chose de compréhensible pour la prochaine couche d'Encoder ou le Decoder.

### Decoder Layer

Le but du Decoder est de générer des séquences de texte. Le Decoder s'arrête de générer du texte lorsqu'il génère un token "end". On peut également mettr en place plusieurs Decoder, ce qui peut booster sa puissance de prédiction.

Les étapes d'**Output Embedding** et de **Positional Encoding** sont les mêmes que pour l'Encoder.

La liste d'inputs du Decoder ne contient à la base qu'un token "start".

#### Masked Multi-Headed Attention

Le but de cette sous couche, comme dans celle de l'Encoder est de produire en sortie des scores d'attention pour les entrées de la couche Decoder. Son fonctionnement diffère cependant légèrement.

En effet, le rôle du Decoder est de générer une séquence de texte mot par mot, par conséquent, les mots générés par le Decoder ne peuvent avoir accès qu'aux mots placés avant eux dans la séquence, pas ceux placés après.

On rajoute donc une étape de "masking" qui consiste à ajouter un "Look-Ahead Mask" à la matrice des scores scalé. Pour se faire, on additionne à la matrice des scores une autre matrice triangulaire inférieure de même dimension que la matrice des scores où le triangle est composé de 0 et le reste de valeurs "-inf" (qui signifie moins l'infini), le reste est à 0. On obtient donc en résultat une matrice **Masked Scores** dont le triangle supérieur est égal à moins l'infini. Une fois que nous aurons appliqué la fonction **softmax** sur cette matrice Masked Scores, toutes les valeurs égales à moins l'infini vont être égales à 0 (car exp(-inf) -> 0). Le Decoder ne portera donc aucune importance sur les mots pas encore générés dans la séquence de texte puisque leur score d'attention sera égal à 0.

#### Deuxième Multi-Headed Attention

Le rôle de cette sous couche est de mapper les inputs de l'Encoder et du Decoder. Le Decoder va décider sur quels inputs de l'Encoder il est judicieux de porter de l'attention. La sortie de cette sous couche va ensuite passer dans une couche Feed Forward.

### Sortie du Decoder Layer

La première couche **Linear** en sortie du Decoder est une couche qui permet de classifier. En sortie de cette couche on obtient un vecteur de taille N où chaque case correspond à une classe. Si par exemple on a 20 classes pour 20 mots, alors la taille du vecteur en sortie de cette couche sera de taille 20.

Ensuite, la couche **softmax** produit des probabilités pour chaque classe. On prend ensuite la classe avec la probabilité la plus élevée et ça sera le mot prédit.

Le Decoder prend ensuite ce mot prédit et l'ajoute à la liste de ses inputs et continue de prédire jusqu'à ce que le token "end" soit prédit. Ce token est produit quand c'est la dernière classe du vecteur qui a la probabilité la plus élevée.

## Vision Transformer (ViT)

Les ViT ont été conçus pour traiter des images de manière plus efficace et avec moins de données d'entraînement en utilisant une approche basée sur les Transformers.

#### Fonctionnement du ViT

Le fonctionnement des ViT est assez simple. Tout d'abord, l'image est divisée en une grille de petites images carrées appelées "patches". Chaque patch est ensuite linéarisé et transmis à une couche d'attention, qui calcule les poids d'attention pour chaque patch en fonction de ses relations avec les autres patches.

Les poids d'attention sont ensuite utilisés pour agréger les informations de chaque patch en un seul vecteur de caractéristiques, qui est transmis à une série de couches de transformation linéaire pour produire une représentation de l'image.

Les ViT peuvent être utilisés pour la segmentation d'images en utilisant une approche appelée "segmentation basée sur l'attention" ou "attention-based segmentation". Dans cette approche, les ViT sont utilisés pour extraire des caractéristiques à partir de l'image, puis une couche d'attention est utilisée pour attribuer une étiquette de segment à chaque pixel de l'image en prenant en compte les relations spatiales entre les pixels.


