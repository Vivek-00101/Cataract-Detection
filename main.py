'''import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.image import decode_image
from tensorflow.image import resize
import cv2
from PIL import Image
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load your pre-trained model
model = load_model('model_vgg19.h5')'''

'''
# Define a function to preprocess the uploaded image
def preprocess_image(file):
    image = decode_image(file.read(), expand_animations=False)
    image = resize(image, [224, 224])
    image = image / 255.0  # Normalize the pixel values
    return np.expand_dims(image, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']

    if image.filename == '':
        return redirect(request.url)

    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join('uploads', filename)
        image.save(image_path)

        return redirect(url_for('predict', image_path=image_path))

@app.route('/predict')
def predict():
    image_path = request.args.get('image_path')
    image = Image.open(image_path)
    image = preprocess_image(image)

    # Make predictions using your model
    prediction = model.predict(image)
    # Assuming binary classification, you can adjust this to your specific use case
    class_labels = ['No Cataract', 'Cataract']
    result = class_labels[np.argmax(prediction)]

    return render_template('result.html', image_path=image_path, result=result)

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create the 'uploads' directory if it doesn't exist
    app.run(debug=True)'''



'''
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def prediction(file):
  result1 = ""
  result2 = ""
  start = time.time()
  image = cv2.imread("./static/inputimages/"+file,0)
  image_test = cv2.resize(image, (size, size), interpolation = cv2.INTER_AREA)
  glcm_test=[]
  images_sift_test=[]
  img_arr_test = np.array(image_test)
  gCoMat = graycomatrix(img_arr_test, [1], [0],256,symmetric=True, normed=True) # Co-occurance matrix
  contrast = graycoprops(gCoMat, prop='contrast')[0][0]
  dissimilarity = graycoprops(gCoMat, prop='dissimilarity')[0][0]
  homogeneity = graycoprops(gCoMat, prop='homogeneity')[0][0]
  energy = graycoprops(gCoMat, prop='energy')[0][0]
  correlation = graycoprops(gCoMat, prop='correlation')[0][0]
  keypoints, descriptors = sift.detectAndCompute(image_test,None)
  descriptors=np.array(descriptors)
  descriptors=descriptors.flatten()
  glcm_test.append([contrast,dissimilarity,homogeneity,energy,correlation])
  glcm_test=np.array(glcm_test)
  images_sift_test.append(descriptors[:2304])
  images_sift_test=np.array(images_sift_test)
  images_sift_glcm_test=np.concatenate((images_sift_test,glcm_test),axis=1)
  if(log_pickle_model.predict(images_sift_glcm_test)==1):
      result1 = "Cataract detected"
      filename = "./static/inputimages/"+file
      img = cv2.imread(filename)
      img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
      frame = img
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      sensitivity = 156
      lower_white = np.array([0,0,255-sensitivity])
      upper_white = np.array([255,sensitivity,255])
      # Threshold the HSV image to get only white colors
      mask = cv2.inRange(hsv, lower_white, upper_white)
      # Bitwise-AND mask and original image
      res = cv2.bitwise_and(frame,frame, mask= mask)
      ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
      circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.5, 100000,param1=80,param2=40,minRadius=0,maxRadius=0)
      x,y,r = 0,0,0
      if circles is not None:
          circles = np.uint16(np.around(circles))
          x,y,r = circles[0][0]
          x=int(x)
          y=int(y)
          r=int(r)
      mask = np.zeros((224,224), np.uint8)
      cv2.circle(mask,(x,y),r,(255,255,255),-1)
      masked_data = cv2.bitwise_and(frame, frame, mask=mask)
      _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
      cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
      x,y,w,h = cv2.boundingRect(cnt[0])

      # Crop masked_data
      crop = masked_data[y:y+h,x:x+w]
      crop = cv2.resize(crop, (224,224), interpolation = cv2.INTER_AREA)
      #preprocess the image
      my_image = preprocess_input(crop)
      crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
      my_image = img_to_array(crop)
      my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
      ans = intensity_model.predict(my_image)
      ans_class = np.argmax(ans)
      classes = ["Mild Cataract","Normal Cataract","Severe Cataract"]
      result2 = classes[ans_class]
  else:
      result1 = "No catarcat"
      result2 = "Normal eye"
  return redirect(url_for('results', result1 = result1, result2 = result2))

@app.route('/result/<result1>/<result2>', methods=['GET', 'POST'])
def results(result1, result2):
  return render_template('result.html', result1 = result1, result2 = result2)

if __name__ == "__main__":
  app.run()
'''









import os
import cv2
import time
from flask import Flask, render_template, request, redirect, url_for

# Add the missing imports here
from tensorflow.keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

size = 224  # Define the image size

# Load your pre-trained models
model = load_model('model_vgg19.h5')


# Define a function to preprocess the uploaded image
def preprocess_image(file):
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = preprocess_input(image)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(request.url)

    image = request.files['image']

    if image.filename == '':
        return redirect(request.url)

    if image:
        filename = secure_filename(image.filename)
        image_path = os.path.join('uploads', filename)
        image.save(image_path)

        return redirect(url_for('prediction', file=filename))

@app.route('/predict/<file>')
def prediction(file):
    result1 = ""
    result2 = ""
    start = time.time()
    image = cv2.imread("./uploads/"+file, 0)
    image_test = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    glcm_test = []
    images_sift_test = []
    img_arr_test = np.array(image_test)
    gCoMat = graycomatrix(img_arr_test, [1], [0], 256, symmetric=True, normed=True)  # Co-occurrence matrix
    contrast = graycoprops(gCoMat, prop='contrast')[0][0]
    dissimilarity = graycoprops(gCoMat, prop='dissimilarity')[0][0]
    homogeneity = graycoprops(gCoMat, prop='homogeneity')[0][0]
    energy = graycoprops(gCoMat, prop='energy')[0][0]
    correlation = graycoprops(gCoMat, prop='correlation')[0][0]
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_test, None)
    descriptors = np.array(descriptors)
    descriptors = descriptors.flatten()
    glcm_test.append([contrast, dissimilarity, homogeneity, energy, correlation])
    glcm_test = np.array(glcm_test)
    images_sift_test.append(descriptors[:2304])
    images_sift_test = np.array(images_sift_test)
    images_sift_glcm_test = np.concatenate((images_sift_test, glcm_test), axis=1)

    # Replace 'log_pickle_model' with your loaded model
    if model.predict(images_sift_glcm_test) == 1:
        result1 = "Cataract detected"
        filename = "./uploads/" + file
        img = cv2.imread(filename)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        frame = img
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sensitivity = 156
        lower_white = np.array([0, 0, 255 - sensitivity])
        upper_white = np.array([255, sensitivity, 255])
        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)
        ret, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)
        circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.5, 100000, param1=80, param2=40, minRadius=0, maxRadius=0)
        x, y, r = 0, 0, 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            x, y, r = circles[0][0]
            x = int(x)
            y = int(y)
            r = int(r)
        mask = np.zeros((224, 224), np.uint8)
        cv2.circle(mask, (x, y), r, (255, 255, 255), -1)
        masked_data = cv2.bitwise_and(frame, frame, mask=mask)
        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        cnt = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        x, y, w, h = cv2.boundingRect(cnt[0])

        # Crop masked_data
        crop = masked_data[y:y + h, x:x + w]
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
        # preprocess the image
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        my_image = img_to_array(crop)
        my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
        ans = intensity_model.predict(my_image)
        ans_class = np.argmax(ans)
        classes = ["Mild Cataract", "Normal Cataract", "Severe Cataract"]
        result2 = classes[ans_class]
    else:
        result1 = "No cataract"
        result2 = "Normal eye"

    return render_template('result.html', result1=result1, result2=result2)

@app.route('/result/<result1>/<result2>', methods=['GET', 'POST'])
def results(result1, result2):
    return render_template('result.html', result1=result1, result2=result2)

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run()








