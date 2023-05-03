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

![Softmax formule](https://lh6.googleusercontent.com/3vcfJ5hJhsMZAMFIbQOEycfVW1t6rh1CXt62DeMk8RPPXVzV4vCcURNm_z_F7618uAeSHT7qT7wE_UiK5Ic0b-Eeuunn6iTGeHWbpAaUAP6-G2ePubeGWCb4_TmSapeaimZqvuUs)

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


