import cv2 as cv
import random
haar_cascade = cv.CascadeClassifier('haar_face.xml')
cap = cv.VideoCapture('koshary.mp4')
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    # convert frame to gray
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # detect faces
    faces = haar_cascade.detectMultiScale(gray, 1.1, 10)

    # draw rectangle on the faces
    for(x,y, w,h) in faces:
        cv.rectangle(frame, (x,y), (x+w, y+h), (random.randint(0,255),random.randint(0,255),random.randint(0,255)), thickness= 3)
        # Display the resulting frame
        cv.imshow('frame', frame)
    
    # close video if press q
    if cv.waitKey(1) == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()