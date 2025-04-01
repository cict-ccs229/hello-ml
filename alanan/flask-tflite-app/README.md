# Flask TensorFlow Lite Image Upload App

This project is a simple Flask application that allows users to upload images and process them using a TensorFlow Lite model. 

## Project Structure

```
flask-tflite-app
├── app.py                # Main entry point of the Flask application
├── requirements.txt      # Lists the dependencies required for the project
├── templates
│   └── index.html       # HTML template for the web interface
├── static
│   └── styles.css       # CSS styles for the web application
├── models
│   └── model.tflite     # TensorFlow Lite model for inference
└── README.md             # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd flask-tflite-app
   ```

2. **Create a virtual environment (optional but recommended):**
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Flask application:**
   ```
   python app.py
   ```

2. **Open your web browser and go to:**
   ```
   http://127.0.0.1:5000
   ```

3. **Upload an image using the provided interface.**

## License

This project is licensed under the MIT License.