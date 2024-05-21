import cv2
from matplotlib import pyplot as plt
import numpy as np

rgb_target = cv2.cvtColor(cv2.imread("target.jpg"), cv2.COLOR_BGR2RGB)
gray_target = cv2.cvtColor(rgb_target, cv2.COLOR_RGB2GRAY)

feature_extractor = cv2.SIFT_create()

kp_target, desc_target = feature_extractor.detectAndCompute(gray_target, None)

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False): 
    print("Impossibile aprire il video.")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

bf = cv2.BFMatcher()

ret, frame = cap.read()

while ret:    
 
        cv2.imshow('frame',frame)

        a=cv2.waitKey(40) 
        if a!=-1: 
            ret=False 
        else:
            ret,frame=cap.read()
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        feature_extractor = cv2.SIFT_create()
        kp_frame , desc_frame = feature_extractor.detectAndCompute(gray_frame, None)

        bf = cv2.BFMatcher() 

        matches = bf.knnMatch(desc_target, desc_frame, k=2)

        good_and_second_good_match_list = []
        for m in matches:
            if m[0].distance/m[1].distance < 0.5:
                good_and_second_good_match_list.append(m)
        good_match_arr = np.asarray(good_and_second_good_match_list)

       

        if len(good_match_arr) > 10:
            cv2.putText(frame, 'Oggetto trovato', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


cap.release()
 
cv2.destroyAllWindows() 