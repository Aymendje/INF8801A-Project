# INF8801A-Project
Applications multimédias

Ce proejt est en python2 et demande les modules skimage, numpy, scipy et opencv

Il existe 3 fichier python differents :

	learn_hog.py -> Permet d'apprendre un  HOG
					Ce program permet de generer les 30 images necessaire à l'aprentissage du mouvement representer par la main.
					À l'appui de la touche 's', il sauvegarde une photo de la main et génére les moments de HOG pour cette photo.
					Pour l'instant, l'utilisation est sequentiel, c'est à dire les 30 images doivent etre prise l'une à la suite de l'autre dans le meme orde : 6x la forme 1, 6x la forme 2, ..., 6x la forme 5, donnat un total de 30 images. Ces images sont sauvegarder dans le dossier ~/hog/ et sont en jpg. Dans le meme dossier existe un fichier xml (qui est en fait un blob binaire) qui peut etre importer sous numpy pour representer un tableau contenant les descripteurs de HOG associés à l'image.

	plot_hog.py ->  Permet de montrer a quoi ressemble un HOG
					C'est un utilitaire facultatif, mais permet de representer en temps reel ce que l'ordinateur voit comme vecteur de direction (descripteurs de HOG) pour le model que la camera filme. Malgres que ce soit en temps reel, il y a un leger delai (du au calcul intense) quand on bouge la main.

	project.py -> 	Prend ce que genere learn_hog et l'interprete.
					C'est le projet en soit, il lit tout les fichiers xml dans le dossier ~/hog/ et compare en temps reel la representation courante de la main avec le contenu de ces descripteurs, et choisi celui à la distance la plus petite (dans une marge de 5%).