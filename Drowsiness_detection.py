import dlib
import cv2
import numpy as np 
 
from imutils import face_utils

#Initializing the camera from opencv
cap  = cv2.VideoCapture(0)

#initializing the face detector and landmark detector
#It is more accurate than opencv therefore we are using dlib library
detector = dlib.get_frontal_face_detector()
#It will detect 68 facial landmarks 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


#different states
sleep=0
drowsy = 0
active = 0
status = ""
color = (0,0,0)

#to calculate distance between two detected points using euclidian distance
def compute(ptA,ptB):
    dist = np.linalg.norm(ptA-ptB)
    return dist


def blinked(a,b,c,d,e,f):
    up = compute(b,d) + compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    if(ratio>0.25):
        return 2
    elif(ratio>0.21 and ratio<=0.25):
        return 1
    else:
        return 0

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    face_frame = frame.copy()
    #detected faces in faces array
    for face in faces:
        #the rectangle which is displayed on screen
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # face_frame = frame.copy()
        cv2.rectangle(face_frame,(x1,y1),(x2,y2),(0,255,0),2)

        #passing the frame(gray) and area of detection(face)
        landmarks = predictor(gray,face)
        #converting the detected landmarks into np array
        landmarks = face_utils.shape_to_np(landmarks)

        #the detected landmarks have numbers in sorted array format
        left_blink = blinked(landmarks[36],landmarks[37],
                landmarks[38],landmarks[41],landmarks[40],landmarks[39])
        right_blink = blinked(landmarks[42],landmarks[43],
                landmarks[44],landmarks[47],landmarks[46],landmarks[45])
        
        #now judging what to do for the eye blinks
        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy=0
            active=0
            if(sleep>6):
                status = "SLEEPING !!!"
                color = (255,0,0)
        elif(left_blink==1 or right_blink==1):
            sleep=0
            active=0
            drowsy+=1
            if(drowsy>6):
                status = "DROWSY !"
                color = (0,0,255)
        else:
            drowsy=0
            sleep=0
            active+=1
            if(active>6):
                status="ACTIVE :)"
                color = (0,255,0)
        
        cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)
        #68 landmarks
        for n in range(0,68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame,(x,y),1,(255,0,0),-1)

    cv2.imshow("frame",frame)
    cv2.imshow("Result of detector",face_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break        





