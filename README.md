Digi P\&ID: AI-Powered P&ID Digitization Platform

 An intelligent web application that automates the digitization of Piping and Instrumentation Diagrams (P\&IDs) using deep learning, and enhances accuracy through a powerful Human-in-the-Loop (HITL) interface.

<img width="1919" height="1199" alt="image" src="https://github.com/user-attachments/assets/388fb886-97e3-40d2-8428-c6981934bf2e" />


 Key Features

  * Automated Symbol & Text Recognition: Utilizes a YOLOv8 model for symbol detection and Doctr/Tesseract OCR for tag and text recognition.
  * Intelligent Graph Construction: Traces pipe connections to build a network graph where symbols are nodes and pipes are edges.
  * Multiple Export Formats: Generates structured data outputs in simple JSON, XML, and the industry-standard ISO 15926 (JSON) format.
  * Interactive Human-in-the-Loop (HITL): A user-friendly interface for engineers to validate, correct, and refine the AI's output.
  * Continuous AI Improvement: All corrections are saved as feedback data, which can be used to fine-tune and improve the underlying AI model over time.
  * Modern & Responsive UI: A clean, mobile-first interface designed for ease of use on any device.

 Technology Stack

  * Backend: Python, Flask, Ultralytics YOLOv8, OpenCV, python-doctr, pytesseract, NetworkX
  * Frontend: HTML5, Tailwind CSS, Vanilla JavaScript

Installation & Setup:
   Follow these steps to get the project running on your local machine.

1.Prerequisites:
  you must have the following installed on your system:

  * Python 3.8+: [Download Python](https://www.python.org/downloads/)
  * Tesseract-OCR: This is crucial for the OCR engine.
      * [Installation Guide for Tesseract](https://tesseract-ocr.github.io/tessdoc/Installation.html)
      * Windows Users: During installation, make sure to add Tesseract to your system's PATH. You may need to update the path in the `app.py` script: `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`
  * Poppler: Required for processing PDF files.
      * [Installation for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
      * macOS: `brew install poppler`
      * Linux: `sudo apt-get install poppler-utils`

2.Project Setup:
```bash
 1. Clone the repository
git clone <your-repository-url>
cd <repository-folder>

 2. Create and activate a Python virtual environment
python -m venv venv
 Windows
venv\Scripts\activate
 macOS / Linux
source venv/bin/activate

 3. Install the required Python packages
pip install -r requirements.txt

 4. Download your trained YOLO model
 Make sure your trained model file is named 'best.pt' and placed in the root of the project folder.
```

3.Create `requirements.txt`

Create a file named `requirements.txt` in your project folder and add the following packages:

```
flask
flask_cors
ultralytics
opencv-python
python-doctr[torch]
pytesseract
networkx
pdf2image
scikit-image
shapely
numpy
```

4.Running the Application

1.  Start the Backend Server:
    Open a terminal, activate your virtual environment, and run the Flask app.

    ```bash
    python app.py
    ```

    The server will start, usually at `http://127.0.0.1:5000`.

2.  Launch the Frontend:
    Open the `index.html` file in your web browser (preferably with a Live Server extension in VS Code for the best experience).

Usage:
1.  Open the application in your browser.
2.  Upload an Image or PDF file using the interface.
3.  Click the "Analyze Document" button.
4.  After processing, you will be taken to the Results page.
5.  Review the digitized diagram, summary, and raw data.
6.  Download the results in your desired format (JSON, XML, ISO).

The Human-in-the-Loop (HITL) Workflow:
  This platform's key strength is the synergy between AI and human expertise.

1.  AI First Pass: The system automatically digitizes the uploaded P\&ID, creating a baseline result.
2.  Human Validation: An engineer reviews the annotated diagram on the Results page.
3.  Correction & Refinement:
      * To correct a tag, simply click on the symbol in the diagram or the "[Edit]" button in the Interactive JSON view.
      * To add a missing symbol, click and drag a box on an empty area of the diagram. A pop-up will ask for the symbol's class name.
4.  Submit Feedback: Click the "Submit New Symbols" button. This saves your corrections to the `feedback_data` folder on the server. If you add a symbol class the AI has never seen, it will prompt you to upload more training images.
5.  Improve the AI: Once enough correction data has been collected, click the "Start Fine-Tuning" button. This triggers a background process to re-train the AI model using your feedback, making it more accurate for future documents.

Project Structure:
```
.
├── app.py               The main Flask backend server
├── finetune.py          Script for re-training the YOLO model
├── index.html             The complete frontend application
├── best.pt                Your trained YOLOv8 model file
├── class_names.json     List of symbol classes
├── requirements.txt       Python dependencies
└── feedback_data/         Directory for storing HITL corrections
    ├── images/
    ├── labels/
    └── tags/
```

-----
