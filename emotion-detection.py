import cv2 as cv
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

face_detector = cv.CascadeClassifier('haar_face.xml')

# cap = cv.VideoCapture("your_test_video.mp4")

# for camera streaming
cap = cv.VideoCapture(0)

# if there is an issue with a camera
if not cap.isOpened():
    print("Cannot open camera")
    # make a black image
    img = np.zeros((720,1024,3), np.uint8)
    # put text on it
    cv.putText(img, "Sorry, We can't access your camera.", (250, 360),cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)
    # show the image
    cv.imshow("camera Error", img)
    # pause the program
    cv.waitKey(0)
else:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv.resize(frame, (1024, 720))
        # if frame is read correctly ret is True
        
        if not ret:
            print("Can't receive frame. Exiting ...")
            break
        
        # convert frame to gray
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # detect faces available on camera
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # take each face available on the camera and Preprocess it
        for (x, y, w, h) in num_faces:
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)

            # get the max value of the pedection
            maxindex = int(np.argmax(emotion_prediction))

            clr = (255, 0, 0)
            if(maxindex == 0): clr = (0, 0, 255)
            elif(maxindex == 1): clr = (0, 255, 255)
            elif(maxindex == 2): clr = (0, 120, 180)
            elif(maxindex == 3): clr = (200, 20, 20)
            elif(maxindex == 4): clr = (0, 255, 0)
            elif(maxindex == 5): clr = (255, 255, 255)
            elif(maxindex == 6): clr = (100, 100, 100)

            # put a border around each face
            cv.rectangle(frame, (x,y), (x+w, y+h), clr, thickness= 3)

            #put the emotion predection on each face
            cv.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv.FONT_HERSHEY_SIMPLEX, 1, clr, 2, cv.LINE_AA)
            cv.putText(frame, "press Q to exit",(10,20), cv.FONT_HERSHEY_SIMPLEX, .6, (0,0,0), 2, cv.LINE_AA)

        # show the frame
        cv.imshow('Emotion Detection', frame)

        # exit if press 'Q'
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()