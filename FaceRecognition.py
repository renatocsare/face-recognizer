import cv2 as cv
import numpy as np
from keras.models import load_model

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['F', 'M']

#https://learning.oreilly.com/learning-paths/learning-path-computer/9781838824518/9781789950816-video7_5

#load keras model
model = load_model('keras_model/my_model.hdf5')
model.get_config()

target =['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_caffe_models():
    age_net = cv.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return (age_net, gender_net)

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv.CascadeClassifier(cascPath)
video_capture = cv.VideoCapture(0)

age_net, gender_net = load_caffe_models()

def video_detector(age_net, gender_net):
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get Face
        face_img = frame[y:y + h, h:h + w].copy()

        blob = cv.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        # Predict the Emotion
        result = target[np.argmax(model.predict(face_img))]

        overlay_text = "%s %s %s" % (gender, age, result)
        cv.putText(frame, overlay_text, (x, y), cv.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv.LINE_AA)

        # Display the resulting frame
        cv.imshow('Video', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break


video_detector(age_net, gender_net)

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()