from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import face_recognition
import numpy as np
import argparse
import imutils
import pickle
import cv2
import time
import os
#ip = 'rtsp://admin:123456@172.16.12.151'
# Cai dat cac tham so dau vao
ap = argparse.ArgumentParser()
ap.add_argument("-m",
                "--model",
                type=str,
                required=True,
                help="path to trained model")
ap.add_argument("-l",
                "--le",
                type=str,
                required=True,
                help="path to label encoder")
ap.add_argument("-d",
                "--detector",
                type=str,
                required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c",
                "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#####################--RECOGNITION--##################
# Tải ảnh người cần nhận diện (đầu tiên là mình) va ma hoa anh do.

Hoang_image = face_recognition.load_image_file("Hoang.jpg")
Hoang_face_encoding = face_recognition.face_encodings(Hoang_image)[0]


Duong_image = face_recognition.load_image_file("Duong.jpg")
Duong_face_encoding = face_recognition.face_encodings(Duong_image)[0]

SepChien_image = face_recognition.load_image_file("SepChien.jpg")
SepChien_face_encoding = face_recognition.face_encodings(SepChien_image)[0]
# Tạo 1 mang cac khuon mat da duoc ma hoa va ten
known_face_encodings = [
    Hoang_face_encoding,
    Duong_face_encoding
]
known_face_names = [
    "Duc Hoang",
    "Duong",
    "Sep Chien"
]

       
 ############################### Face Mask Detec #################################################
def detect_and_predict_mask(frame, net, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	net.setInput(blob)
	detections = net.forward()
	#print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]            
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
                       

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

            
            
	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# Load model nhan dien khuon mat
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join(
    [args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Load model nhan dien fake/real
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())          
                   

# load our serialized face detector model from disk
# prototxtPath = r"face_detector\deploy.prototxt"
# weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
# faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    frame = vs.read()
     
    # to have a maximum width of 400 pixels  
    frame = imutils.resize(frame, width=700)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Chuyen thanh blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, net, maskNet)

    net.setInput(blob)
    detections = net.forward()

    # Loop qua cac khuon mat
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
# Neu conf lon hon threshold
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

           # Lay vung khuon mat
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

           # Dua vao model de nhan dien fake/real 
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # Ve hinh chu nhat quanh mat
            preds_ln = model.predict(face)[0]
            j = np.argmax(preds_ln)
            label = le.classes_[j]
            if (j==1):  #Nếu khuôn mặt real j=1 và sẽ có box màu đỏ
               # print('real Face')
                label = "Real Face".format(label, preds_ln[j])
                cv2.putText(frame, label, (startX, startY - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
                    (startX, startY, endX, endY) = box
                    (mask, withoutMask) = pred
                    label = "Mask" if mask > withoutMask else "No Mask"
                    color = (0, 254, 0) if label == "Mask" else (0, 0, 254)
                    #print(label)
            # include the probability in the label
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) 

            # display the label and bounding box rectangle on the output
            # frame
                    cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    name = "Unknown"
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    cv2.rectangle(frame, (startX, endY - 35), (endX, endY), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (startX + 6, endY - 6), font, 1.0, (0, 0, 0), 1)


            else:
                

                label = "{}: {:.4f}".format(label, preds_ln[j])
                cv2.putText(frame, label, (startX, startY - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255),
                          2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"):         
        break
cv2.destroyAllWindows()
vs.stop()

  
