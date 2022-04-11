import cv2
import imutils
# LAU 6
ip = 'rtsp://admin:123456@172.16.12.151'

# LAU 7
#ip = 'rtsp://admin:LacV!3t$@192.168.100.83:554/Streaming/Channels/101/'

cap = cv2.VideoCapture(ip)
while True:
    flag, frame = cap.read()
    try:
        cv2.imshow('Camera IP', frame)
    except:
        cap.release()
        raise

    if cv2.waitKey(5) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
