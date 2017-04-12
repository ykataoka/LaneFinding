import numpy as np
import cv2

cap = cv2.VideoCapture('challenge_video3.mp4')
i = 0

while(cap.isOpened()):
    ret, frame = cap.read()

    if i % 2 == 0:
        outname = 'test_challenge3/' + str(int(i)) + '.jpg'
        print(outname)
        cv2.imwrite(outname, frame)
    i += 1
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#        cv2.imshow('frame',gray)
#        if cv2.waitKey(1) & 0xFF == ord('q'):
#            break

cap.release()
cv2.destroyAllWindows()
