from skimage.feature import hog
from skimage import color, exposure

import cv2

cap = cv2.VideoCapture(0)


ret, image_1 = cap.read() 
#image_1 = cv2.blur(image_1, (15,15))
while(1):
    ret, image_2 = cap.read()
    #image_2 = cv2.blur(image_2, (15,15))
    image = cv2.subtract(image_1,image_2)
    image_1 = image_2
    cv2.imshow('P', image)

    image = color.rgb2gray(image)
    image = cv2.blur(image, (5,5))

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise=True)
    
    

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.01))

    cv2.imshow('Histogram of Oriented Gradients', hog_image_rescaled)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
