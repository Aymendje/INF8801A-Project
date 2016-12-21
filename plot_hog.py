# initialisations
from skimage.feature import hog
from skimage import color, exposure
import cv2

cap = cv2.VideoCapture(0)
ret, image_1 = cap.read() 

# On lit en permanence depuis la camera
while(1):
    ret, image_2 = cap.read()
    # On soustrait l'image presente a la precedente (voir "learn_hogs.py")
    image = cv2.subtract(image_1,image_2)
    image_1 = image_2
    cv2.imshow('P', image)

    image = color.rgb2gray(image)
    image = cv2.blur(image, (5,5))

    # On calcul les descripteurs de hog que l'on met dans un histogram
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    
    

    # On rescale l'histogram pour mieux voir
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.01))

    # Et on l'affiche
    cv2.imshow('Histogram of Oriented Gradients', hog_image_rescaled)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
