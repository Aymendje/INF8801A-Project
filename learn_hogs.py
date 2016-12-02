import numpy as np
import cv2
video = cv2.VideoCapture(0)
hog_desc = cv2.HOGDescriptor()

ret, image_1 = video.read()

fileNames = ["5_0", "5_1", "5_2", "5_3", "5_4", "5_5",
            "4_0", "4_1", "4_2", "4_3", "4_4", "4_5",
            "3_0", "3_1", "3_2", "3_3", "3_4", "3_5",
            "2_0", "2_1", "2_2", "2_3", "2_4", "2_5",
            "1_0", "1_1", "1_2", "1_3", "1_4", "1_5",
            "0_0", "0_1", "0_2", "0_3", "0_4", "0_5"]

for fileName in fileNames:
    print(fileName) 
    while(True):
        # Capture frame-by-frame
        ret, frame = video.read()
        image = cv2.subtract(image_1,frame)
        #image_1 = frame

        # Our operations on the frame come here
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(display,(300,300),(100,100),(0,255,0),0)
        display = cv2.resize(display,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('frame',display)
        crop_img = gray[100:300, 100:300]
        
        hog = hog_desc.compute(crop_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            fileName = "./hog/"+fileName
            hog.tofile(fileName+".xml")
            cv2.imwrite(fileName+".jpg", crop_img)
            break
        elif key == 27:
            video.release()
            cv2.destroyAllWindows()
            exit(0)


# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
