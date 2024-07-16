# Skin Disease Classifier

This web application uses a computer vision model to classify skin disease images uploaded by users or queried by disease name. It leverages Flask for the backend and EfficientNet for image classification.
Images from the app are not shown due to their graphic nature but I do invite you to test the app on your own. The datset used for this project comes from kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
## Features

- **Upload Image:** Users can upload an image of a skin disease, and the application will classify it into one of seven categories.
  
- **Query by Disease Name:** Users can enter a disease name to retrieve and view images associated with that disease.

## Prerequisites

- Python 3.x
- Flask
- Torch
- EfficientNet (Python package for EfficientNet model)
- PIL (Python Imaging Library)
- Other dependencies (see \`requirements.txt\`)

## Installation

1. Clone the repository:

   \`\`\`bash
   git clone https://github.com/your/repository.git
   cd repository
   \`\`\`

2. Install dependencies:

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. Download pre-trained EfficientNet model weights (if required) and place them in the appropriate directory.

## Usage

1. Start the Flask application:

   \`\`\`bash
   python app.py
   \`\`\`

2. Open your web browser and go to \`http://localhost:5000\` to access the application.

3. **Upload Image:**
   - Click on "Upload Image" and select an image file containing a skin disease.
   - Click "Submit" to classify the disease.
   
4. **Query by Disease Name:**
   - Enter a disease name in the search box.
   - Click "Search Images" to view images associated with the disease.

## Directory Structure

\`\`\`
project/
│
├── app.py                    # Flask application entry point
├── static/                   # Directory for static assets (images)
│   ├── disease1/
│   ├── disease2/
│   └── ...
├── templates/                # HTML templates
│   ├── index.html            # Main page template
│   └── images.html           # Template for displaying images by disease
├── Environment.py            # Configuration file or module
├── model.pth                 # Pre-trained model weights (if applicable)
└── README.md                 # Project documentation
\`\`\`

## Notes

- Ensure that the model (\`model.pth\` or similar) and any required configurations (\`Environment.py\`) are correctly set up before running the application.
- Modify paths and configurations as necessary based on your environment and requirements.

## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests.


