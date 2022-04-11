import cv2

#print("Before URL")

ip = 'rtsp://admin:LacV!3t$@192.168.100.83:554/Streaming/Channels/101/'

cap = cv2.VideoCapture(ip)
#print("After URL")

while True:

    #print('About to start the Read command')
    ret, frame = cap.read()
    #print('About to show frame of Video.')
    cv2.imshow("Capturing",frame)
    #print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()