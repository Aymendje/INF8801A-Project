# initialisations
import numpy as np
import cv2

video = cv2.VideoCapture(0)
hog_desc = cv2.HOGDescriptor()

ret, image_1 = video.read()

# Tableau des 30 images à sauvegarder.
fileNames = ["5_0", "5_1", "5_2", "5_3", "5_4", "5_5",
            "4_0", "4_1", "4_2", "4_3", "4_4", "4_5",
            "3_0", "3_1", "3_2", "3_3", "3_4", "3_5",
            "2_0", "2_1", "2_2", "2_3", "2_4", "2_5",
            "1_0", "1_1", "1_2", "1_3", "1_4", "1_5",
            "0_0", "0_1", "0_2", "0_3", "0_4", "0_5"]

# Pour chaque image à sauvegarder, on effectue toute les opérations :
for fileName in fileNames:
    print(fileName) 
    while(True):
        # On va chercher la premiere image
        ret, frame = video.read()

        # On soustrait les differences entre l'image courante et la précédente
        # (on cherche uniquement ce qui a bouger, ce qui est "different")
        # Dans notre cas ici, vu que c'est une main humaine, elle bouge tout le temps
        image = cv2.subtract(image_1,frame)

        # On transforme en noir et blanc
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # On montre a l'utilisateur ce que la webcam voit (donc en couleur, sans la soustraction)
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # On rajoute un carré de 100x100 pour montrer quel endroti va etre analyser par le program
        # C'est là qu'il faut mettre la main
        cv2.rectangle(display,(300,300),(100,100),(0,255,0),0)
        display = cv2.resize(display,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('frame',display)

        # On gree le descripteur de hog
        crop_img = gray[100:300, 100:300]
        hog = hog_desc.compute(crop_img)

        # On enregistre l'image et son descripteur, puis on passe à la prochaine.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            fileName = "./hog/"+fileName
            hog.tofile(fileName+".xml")
            cv2.imwrite(fileName+".jpg", crop_img)
            break
        elif key == 27:
            # Si on appui sur ESCAPE, on quitte tout
            video.release()
            cv2.destroyAllWindows()
            exit(0)


# On relache la capture
video.release()
cv2.destroyAllWindows()
