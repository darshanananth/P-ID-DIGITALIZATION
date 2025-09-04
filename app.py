# ==============================================================================
# FINAL, PRODUCTION-READY BACKEND SERVER (v3 - with JSON Output)
# ==============================================================================
print("--- Initializing Backend Server ---")

import os
import re
import json
import base64
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
from ultralytics import YOLO
from doctr.models import ocr_predictor
import pytesseract
from skimage.morphology import skeletonize
from collections import defaultdict, deque
import traceback

# --- Initialize Flask App ---
app = Flask(__name__)
CORS(app)

# --- Load AI Models ---
print("Loading AI Models... This may take a moment.")
try:
    model_path = Path("best.pt")
    if not model_path.is_file():
        raise FileNotFoundError(f"Could not find 'best.pt'. Make sure it's in the same folder as app.py.")
    symbol_detector = YOLO(model_path)
    text_detector = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True, export_as_straight_boxes=True)
    print("✅ All AI models loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load AI models: {e}")
    exit()

@app.route('/digitize', methods=['POST'])
def digitize_pid():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # --- 1. CONVERT INPUT FILE TO AN IMAGE ---
        file_bytes = file.read()
        if file.filename.lower().endswith('.pdf'):
            images = convert_from_bytes(file_bytes, first_page=1, last_page=1, poppler_path=r'C:\Users\D_SHAAN\Release-25.07.0-0\poppler-25.07.0\Library\bin')
            if not images: raise ValueError("Could not process PDF.")
            original_img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        else:
            original_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        # --- 2. RUN THE FULL DIGITIZATION PIPELINE ---
        
        # Stage A: Symbol Detection
        results_yolo = symbol_detector(original_img, verbose=False)[0]
        yolo_data = [{"bbox": tuple(map(int, box.xyxy[0])), "class_name": symbol_detector.names[int(box.cls.item())]} for box in results_yolo.boxes if box.conf.item() > 0.4]

        # Stage B: OCR
        gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        preprocessed_ocr = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        doctr_results = text_detector([original_img])
        word_boxes = [w.geometry for b in doctr_results.pages[0].blocks for l in b.lines for w in l.words]
        ocr_results_list = []
        for box in word_boxes:
            x1, y1, x2, y2 = int(box[0][0] * original_img.shape[1]), int(box[0][1] * original_img.shape[0]), int(box[1][0] * original_img.shape[1]), int(box[1][1] * original_img.shape[0])
            crop = preprocessed_ocr[y1:y2, x1:x2]
            if crop.size > 0:
                text = pytesseract.image_to_string(crop, config=r'--oem 3 --psm 7').strip()
                if text: ocr_results_list.append({"text": text, "bbox": (x1, y1, x2, y2)})
        
        # Stage C: Line Detection & Road Map
        _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        for symbol in yolo_data: cv2.rectangle(binary, symbol["bbox"][:2], symbol["bbox"][2:], (0,0,0), -1)
        skeleton = skeletonize(binary // 255).astype(np.uint8)

        # Stage D: Graph Construction
        graph = {}
        for i, symbol in enumerate(yolo_data):
            node_id = i + 1
            x1, y1, x2, y2 = symbol["bbox"]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            nonzero_y, nonzero_x = np.nonzero(skeleton)
            if nonzero_y.size == 0: continue
            distances = np.sqrt((nonzero_x - cx)**2 + (nonzero_y - cy)**2)
            on_ramp_point = (nonzero_y[np.argmin(distances)], nonzero_x[np.argmin(distances)])
            best_tag = min(ocr_results_list, key=lambda ocr: np.sqrt((cx - (ocr["bbox"][0]+ocr["bbox"][2])/2)**2 + (cy - (ocr["bbox"][1]+ocr["bbox"][3])/2)**2), default={}).get("text")
            graph[node_id] = {"id": node_id, "type": symbol["class_name"], "tag": best_tag, "coords": (cx, cy), "bbox": symbol["bbox"], "on_ramp": on_ramp_point, "connections": []}
        
        for start_node_id, start_data in graph.items():
            q, visited = deque([start_data["on_ramp"]]), {start_data["on_ramp"]}
            while q:
                y, x = q.popleft()
                for end_node_id, end_data in graph.items():
                    if start_node_id != end_node_id and (y, x) == end_data["on_ramp"] and end_node_id not in start_data["connections"]:
                        start_data["connections"].append(end_node_id)
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx] == 1 and (ny, nx) not in visited:
                        visited.add((ny, nx)); q.append((ny, nx))
        
        # --- 3. PREPARE AND RETURN FINAL OUTPUT ---
        final_img = original_img.copy()
        final_img[skeleton == 1] = [200, 200, 200]
        # Create a list of dictionaries for the final JSON output
        graph_data_json = []
        for node_id, data in graph.items():
            x1, y1, x2, y2 = data["bbox"]; cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(final_img, str(node_id), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,0), 3)
            connections = sorted(list(set(data['connections'])))
            # Add node data to our JSON list
            graph_data_json.append({"node_id": node_id, "type": data['type'], "tag": data['tag'], "connections": connections})
            for neighbor_id in connections:
                if node_id < neighbor_id: cv2.line(final_img, data["coords"], graph[neighbor_id]["coords"], (0, 255, 0), 3)
        
        _, buffer = cv2.imencode('.jpg', final_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({"annotated_image": f"data:image/jpeg;base64,{img_base64}", "graph_data": graph_data_json})

    except Exception as e:
        print(f"An error occurred during digitization: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

