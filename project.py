import numpy as np
import cv2
video = cv2.VideoCapture("Typing3.mp4")
hog_desc = cv2.HOGDescriptor()

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
#   h = cv2.resize(h,None,fx=4, fy=2, interpolation = cv2.INTER_CUBIC)
    # Display histogram
    cv2.imshow('Histogram2', h)

while(True):
    # Capture frame-by-frame
    ret, frame = video.read()

    rows,cols,color = frame.shape
#    M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
#    frame = cv2.warpAffine(frame,M,(cols,rows))

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)


    #show_histogram_color(frame)
    #show_histogram_black(frame)
    
    hog = hog_desc.compute(frame)
    #cv2.imshow("Hog", hog)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
video.release()
cv2.destroyAllWindows()
