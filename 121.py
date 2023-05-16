import cv2
import mediapipe as mp

video=cv2.VideoCapture(0)

myhands=mp.solutions.hands

mydrawing=mp.solutions.drawing_unils

hand_object=myhands.Hands(mim_detection_confidence=0.75,mim_tracking_confidence=0.75)


while True:
    dummy,frame= video.read()
    flipImage=cv2.flip(frame,1)

    result=hand_object.process(cv2.cvtColor(flipImage,cv2.COLOR_BGR2RGB))
    # print(result)

    if result.multi_hand_landmarks:
        hand_keypoints=result.multi_hand_landmarks[0]

        print(hand_keypoints)
        mydrawing.draw_landmarks(flipImage,hand_keypoints,myhands.HAND_CONNECTIONS)


    cv2.imshow("hand gesture", frame)
    if cv2.waitKey(25)==32:
        break

video.release()
cv2.destroyAllWindows()
