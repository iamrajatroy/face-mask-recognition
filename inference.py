##################################
'''
Model Prediction code for Mask Detection
'''
##################################
import warnings
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import config as cfg


def get_predicted_label(model, img, IMG_WIDTH, IMG_HEIGHT):
	img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
	img = img/255
	img = np.expand_dims(img, 0)
	prediction = model.predict(img)
	result = np.argmax(prediction)
	return result

def predict():
	IMG_WIDTH = cfg.IMG_WIDTH
	IMG_HEIGHT = cfg.IMG_HEIGHT
	model_file_path = cfg.MODELS_PATH + 'MODEL_20210427_224637.h5'
	model = load_model(model_file_path)
	labels_dict={0:'NOT WEARING A MASK', 1:'WEARING A MASK'}
	face_cascade = cv2.CascadeClassifier(cfg.MODELS_PATH + 'haarcascade_frontalface_default.xml')
	color_dict={0:(0,0,255), 1:(0,255,0)}
	source = cv2.VideoCapture(0)
	# IP Webcam address
	address = 'http://192.168.1.6:8080/video'
	source.open(address)
	while True:
		_,img = source.read()
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		face_boundary = face_cascade.detectMultiScale(gray_img, 1.1, 5, cv2.CASCADE_SCALE_IMAGE)
		for x,y,w,h in face_boundary:
			cropped_face = img[y:y+h, x:x+w]
			result = get_predicted_label(model, cropped_face, IMG_WIDTH, IMG_HEIGHT)
			cv2.rectangle(img, (x,y),(x+w,y+h), color_dict[result], 2)
			cv2.putText(img, labels_dict[result], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dict[result],2)
		# print(labels_dict[result])
		cv2.imshow('Face Mask Detection App', img)
		key=cv2.waitKey(1)
		if(key==27):
			break
	cv2.destroyAllWindows()
	source.release()


if __name__ == '__main__':
	warnings.filterwarnings('ignore')
	config = tf.compat.v1.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=config)
	predict()