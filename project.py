import numpy as np
import cv2, os, collections
from scipy.spatial import distance
video = cv2.VideoCapture(0)
hog_desc = cv2.HOGDescriptor()

hog_list = []

def show_histogram_color(frame):
    # Now create a histogram for the frame
    h = np.zeros((300, 256, 3))
    b, g, r = cv2.split(frame)
    bins = np.arange(256).reshape(256, 1)
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for item, col in zip([b, g, r], color):
        hist_item = cv2.calcHist([item], [0], None, [256], [0, 255])
        cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
        hist = np.int32(np.around(hist_item))
        pts = np.column_stack((bins, hist))
        cv2.polylines(h, [pts], False, col)

    h = np.flipud(h)
#   h = cv2.resize(h,None,fx=4, fy=2, interpolation = cv2.INTER_CUBIC)
    # Display histogram
    cv2.imshow('Histogram', h)


def show_histogram_black(frame):
    # Now create a histogram for the frame
    h = np.zeros((300, 256, 1))
    bins = np.arange(256).reshape(256, 1)

    hist_item = cv2.calcHist([frame], [0], None, [256], [0, 255])
    cv2.normalize(hist_item, hist_item, 0, 255, cv2.NORM_MINMAX)
    hist = np.int32(np.around(hist_item))
    pts = np.column_stack((bins, hist))
    cv2.polylines(h, [pts], False, 1)

    h = np.flipud(h)
    # h = cv2.resize(h,None,fx=4, fy=2, interpolation = cv2.INTER_CUBIC)
    # Display histogram
    cv2.imshow('Histogram2', h)

def load_learned_numbers():
    # dtype=float32
    global hog_list
    for file in os.listdir("./hog"):
        if file.endswith(".xml"):
            hog_list.append([file, np.fromfile("./hog/"+file, dtype=np.float32)])


def compare_hogs(my_hog):
    global hog_list
    closest = ["Name", float('inf')]
    for pair in hog_list:
        if(len(my_hog) != len(pair[1])): 
            print("HOG of different sizes " + str(len(my_hog)) + " != " + str(len(pair[1])))
        else:
            match = distance.euclidean(my_hog,pair[1])
            if match < closest[1]:
                closest[0] = pair[0]
                closest[1] = match
    return closest

load_learned_numbers()
prepare_hog = 0
error_hog = []
ret, image_1 = video.read()
guess = collections.deque(maxlen=10)
while(True):
    # Capture frame-by-frame
    ret, frame = video.read()
    image = cv2.subtract(image_1,frame)
    #image_1 = frame


    # Our operations on the frame come here
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    show_histogram_color(frame[100:300, 100:300])
    show_histogram_black(frame[100:300, 100:300])

    # Display the resulting frame
    cv2.rectangle(frame,(300,300),(100,100),(0,255,0),0)
    frame = cv2.resize(frame,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('frame',frame)
    crop_img = gray[100:300, 100:300]
    
    hog = hog_desc.compute(crop_img)
    hog = compare_hogs(hog)

    if prepare_hog < 100:
        cv2.putText(frame,str("Loading : " + str(prepare_hog*100/100) + "%"),(200,150), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2)
        error_hog.append(hog[1])
        prepare_hog += 1
        print(hog[1])
    elif prepare_hog == 100:
        error_hog = np.mean(error_hog)
        prepare_hog += 1
        print("Ready !")
    else:
        if hog[1] > error_hog + error_hog/5.0 or hog[1] < error_hog - error_hog/5.0:
            guess.append(hog[0][0])
        else:
            guess.append("NaN")

        cv2.putText(frame,str(max(set(guess), key=guess.count)),(200,200), cv2.FONT_HERSHEY_SIMPLEX, 3,(255,255,255),2)
    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
