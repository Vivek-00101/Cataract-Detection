from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('model_vgg19.h5')



# Define a function to preprocess the uploaded image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        # If the user submits an empty form
        if file.filename == '':
            return "No selected file"

        # If the file is an image, make a prediction
        if file:
            img_path = "uploaded_image.jpg"
            file.save(img_path)
            processed_img = preprocess_image(img_path)
            prediction = model.predict(processed_img)

            # You can adjust the threshold for cataract detection based on your model's output
            is_cataract = prediction[0][0] > 0.5

            if is_cataract:
                result = "You have cataract."
            else:
                result = "You do not have cataract."

            return result

if __name__ == '__main__':
    app.run(debug=True)
