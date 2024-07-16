import io
import torch
from PIL import Image
from efficientnet_pytorch import EfficientNet
from Environment import transform, model_path
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

# Load the trained model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes = 7)
model.load_state_dict(torch.load(model_path))
model.eval()

@app.route('/')
def home():
    
    # Set up simple default page powered by the 'index.html' file.
    return render_template('index.html')

@app.route('/upload', methods = ['POST'])
def upload():
    # Upload images to get a diagnosis.
    file = request.files['file']
    if file:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            class_id = predicted.item()
            # Map class_id to disease name
            disease_name = get_disease_name(class_id)

        return jsonify({'disease_name': disease_name})
    return 'No file uploaded', 400

@app.route('/query', methods=['GET'])
def query():
    # Function to retrieve images based on disease name
    disease_name = request.args.get('disease')
    # Fetch images for the given disease name
    image_urls = get_images_for_disease(disease_name)
    return render_template('images.html', disease_name=disease_name, image_paths=image_urls)


@app.route('/disease-prompt', methods = ['GET'])
def get_disease_name(class_id):
    # Translate class_id from the model to the disease name
    label_map = {
        0: 'Melanocytic nevi',
        1: 'Melanoma',
        2: 'Benign keratosis-like lesions',
        3: 'Basal cell carcinoma',
        4: 'Actinic keratoses',
        5: 'Vascular lesions',
        6: 'Dermatofibroma'
    }
    return label_map[class_id]

def get_images_for_disease(disease_name):
    
    
    if disease_name.lower() == 'nv' or disease_name.lower() == 'melanocytic nevi':
        image_paths = ['nv/ISIC_0024306.jpg','nv/ISIC_0024307.jpg','nv/ISIC_0024308.jpg']
    
    elif disease_name.lower() == 'mel' or disease_name.lower() == 'melanoma':
        image_paths = ['mel/ISIC_0024310.jpg','mel/ISIC_ISIC_0024313.jpg','mel/ISIC_0024315.jpg']

    elif disease_name == 'bkl' or disease_name.lower() == 'benign keratosis-like lesions':
        image_paths = ['bkl/ISIC_0024312.jpg','bkl/ISIC_ISIC_0024324.jpg','bkl/ISIC_0024336.jpg']

    elif disease_name == 'bcc':
        image_paths = ['bcc/ISIC_0024331.jpg','bcc/ISIC_0024332.jpg','bcc/ISIC_0024345.jpg']
    
    elif disease_name.lower() == 'akiec' or 'actinic keratoses':
        image_paths = ['akiec/ISIC_0024329.jpg','akiec/ISIC_0024372.jpg','akiec/ISIC_0024418.jpg']

    elif disease_name.lower() == 'vasc' or 'vascular lesions':
        image_paths = ['vasc/ISIC_0024370.jpg','vasc/ISIC_0024375.jpg','vasc/ISIC_0024402.jpg']
    
    elif disease_name.lower() == 'df' or 'Dermatofibroma':
        image_paths = ['df/ISIC_0024318.jpg','df/ISIC_0024330.jpg','df/ISIC_0024386.jpg']
   
    return image_paths


if __name__ == '__main__':
    app.run(debug=True)