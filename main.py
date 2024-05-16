

from imutils import face_utils
from pygame import mixer
from scipy.spatial import distance
from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
from multiprocessing import Value
import numpy as np
import argparse
import imutils
import time
import cv2
import dlib



def overlap(startX1, startY1, endX1, endY1, startX2, startY2, endX2, endY2, previousTime, currentTime):
    if currentTime - previousTime > 5:
        return False
    hoverlaps = (startX1 <= endX2) and (endX1 >= startX2)
    voverlaps = (startY1 <= endY2) and (endY1 >= startY2)
    return hoverlaps and voverlaps


def classify_frame(net, inputQueue, outputQueue):
    while True:
        if not inputQueue.empty():
            frame = inputQueue.get()
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            outputQueue.put(detections)


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("files/deploy.prototxt.txt", "files/res10_300x300_ssd_iter_140000.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
predict = dlib.shape_predictor("files/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None
count = Value('i', 0)
peopleInLastFrame = 0
lastDetection = None
lastDetectionTime = None
startTime = Value('d', time.time())

print("[INFO] starting face detection process...")
p = Process(target=classify_frame, args=(net, inputQueue, outputQueue,))
p.daemon = True
p.start()

mixer.init()
mixer.music.load("files/music.wav")
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear


# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()
sleepFrameCount = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    (h, w) = frame.shape[:2]

    if inputQueue.empty():
        inputQueue.put(frame)

    if not outputQueue.empty():
        detections = outputQueue.get()

    peopleInThisFrame = 0

    if detections is not None:
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < 0.25:
                continue

            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            isOverlapping = False
            if lastDetection is not None:
                (lastStartX, lastStartY, lastEndX, lastEndY) = lastDetection
                isOverlapping = overlap(startX, startY, endX, endY, lastStartX, lastStartY, lastEndX, lastEndY,
                                        lastDetectionTime, time.time())
                subjects = dlib.rectangles()
                subjects.append(dlib.rectangle(int(startX-5), int(startY-5), int(endX+5), int(endY+5)))
                for subject in subjects:
                    shape = predict(gray, subject)
                    shape = face_utils.shape_to_np(shape)
                    # for i in shape:
                    #     cv2.circle(frame, (i[0], i[1]), 2, (0, 255, 0), cv2.FILLED)
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]

                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    for i in range(len(leftEye)):
                        # cv2.circle(frame, (leftEye[i][0], leftEye[i][1]), 2, (0, 255, 0), cv2.FILLED)
                        cv2.putText(frame, str(i), (leftEye[i][0], leftEye[i][1]), cv2.FONT_HERSHEY_SIMPLEX,  0.4, (0, 255, 0), 2, cv2.LINE_AA)
                    ear = (leftEAR + rightEAR) / 2.0
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                    if ear<0.25:
                        sleepFrameCount += 1
                        if sleepFrameCount > 5:
                            print("sleeping")
                            mixer.music.play()
                    else:
                        sleepFrameCount = 0
                        print("-----------")

            if not isOverlapping:
                peopleInThisFrame = peopleInThisFrame + 1
                if peopleInThisFrame > peopleInLastFrame:
                    with count.get_lock():
                        count.value = count.value + (peopleInThisFrame - peopleInLastFrame)

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            lastDetection = (startX, startY, endX, endY)
            lastDetectionTime = time.time()

    peopleInLastFrame = peopleInThisFrame

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
