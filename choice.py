import cv2
from flask import Flask, render_template, request
import base64
import io
import os
from PIL import Image
import numpy as np

from tensorflow import keras
loaded_model = keras.models.load_model('Conjunctivitis.h5')
gra_model = keras.models.load_model('grading.h5')
image_shape = (120,120,3)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

@app.route('/process_camera', methods=['POST'])
def process_image():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video capture")
        exit()
    # Read the frame from the camera
    ret, frame = cap.read()
    # Check if frame was successfully read
    if not ret:
        print("Error reading frame")
        exit()
    # Save the frame as an image
    cv2.imwrite("captured_image.png", frame)
    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()
    from tensorflow import keras
    loaded_model = keras.models.load_model('Conjunctivitis.h5')
    image_shape = (120,120,3)
    filename = "captured_image.png"
    image = Image.open(filename)
    my_image=  image.resize(image_shape[:2])    
    my_image = np.array(my_image)
    my_image = np.expand_dims(my_image, axis=0)

    result=loaded_model.predict(my_image) # 1 means White Eye and O means Red Eye
    if(result == 1):
        conclusion = "white eye"
        print(conclusion)
    else :
        conclusion = "red eye"
        print(conclusion)

    grad_image_shape = (224, 224)
    grad_image = image.resize(grad_image_shape)
    grad_image = np.array(grad_image)
    grad_image = np.expand_dims(grad_image, axis=0)
    grad_image = grad_image/255.0

    gra_model = keras.models.load_model('grading.h5')
    result = gra_model.predict(grad_image)
    final = result[0]
    new_final = str(final)[6:8] #not able to figure out what this result value is
    print(final,new_final)
    low = '45'
    medium = '70'

    if new_final <= str(low):
        grade_res = "low"
    if new_final >= str(low):
        if new_final <= str(medium):
            grade_res = "medium"
    if new_final >= str(medium):
            grade_res = "high"

    print(grade_res)

    # Encode image as base64 for display in HTML
    with open(filename, "rb") as f:
        img_data = f.read()
        encoded_img = base64.b64encode(img_data).decode('utf-8')
        #print(encoded_img)

    # Render results in a new HTML page
    return render_template('results.html', img_data=encoded_img, conclusion=conclusion, grade_res=grade_res)


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/process_upload', methods=['POST'])
def process_upload():
    # Get uploaded file
    file = request.files['file']
    # Read file data
    img_bytes = file.read()
    # Convert bytes to PIL Image object
    img = Image.open(io.BytesIO(img_bytes))

    # Process image with model
    my_image=  img.resize(image_shape[:2])    
    my_image = np.array(my_image)
    my_image = np.expand_dims(my_image, axis=0)

    result = loaded_model.predict(my_image) # 1 means White Eye and O means Red Eye
    if result == 1:
        conclusion = "white eye"
        print(conclusion)
    else :
        conclusion = "red eye"
        print(conclusion)

    
    grad_image_shape = (224, 224)
    grad_image = img.resize(grad_image_shape)
    grad_image = np.array(grad_image)
    grad_image = np.expand_dims(grad_image, axis=0)
    grad_image = grad_image/255.0
    result = gra_model.predict(grad_image)
    final = result[0]
    new_final = str(final)[6:8] #not able to figure out what this result value is
    print(final,new_final)
    low = '45'
    medium = '70'

    if new_final <= str(low):
        grade_res = "low"
    if new_final >= str(low):
        if new_final <= str(medium):
            grade_res = "medium"
    if new_final >= str(medium):
        grade_res = "high"

    # Save processed image to disk
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = os.path.join(output_dir, 'processed.png')
    img.save(filename)
    
    # Encode image as base64 for display in HTML
    with open(filename, "rb") as f:
        img_data = f.read()
        encoded_img = base64.b64encode(img_data).decode('utf-8')

    print("done")

    # Render results in a new HTML page
    return render_template('results.html', img_data=encoded_img, conclusion=conclusion, grade_res=grade_res)

    
if __name__ == '__main__':
    app.run(debug=True)
