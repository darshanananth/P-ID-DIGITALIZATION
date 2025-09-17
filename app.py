# ==============================================================================
# P&ID DIGITIZATION AI - MASTER BACKEND SERVER (v16 - Full Inconsistency Flagging)
# ==============================================================================
print("--- Initializing Backend Server ---")

import os
import re
import json
import base64
import zipfile
import subprocess
import sys
import time
import threading
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf2image import convert_from_bytes
from ultralytics import YOLO
from skimage.morphology import skeletonize
from collections import defaultdict, Counter, deque
import traceback
import xml.etree.ElementTree as ET
from xml.dom import minidom
from shapely.geometry import LineString, Point
import networkx as nx
from doctr.models import ocr_predictor
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- 1. INITIALIZE FLASK APP & DEFINE PATHS ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit for uploads
CORS(app)

MODEL_PATH = Path("best.pt")
CLASS_NAMES_PATH = Path("class_names.json")
FEEDBACK_DIR = Path("feedback_data")
FEEDBACK_IMAGES_DIR = FEEDBACK_DIR / "images"
FEEDBACK_LABELS_DIR = FEEDBACK_DIR / "labels"
FEEDBACK_TAGS_DIR = FEEDBACK_DIR / "tags"
RESTART_FLAG_FILE = Path("_RESTART_REQUIRED_")

FEEDBACK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_LABELS_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_TAGS_DIR.mkdir(parents=True, exist_ok=True)

# --- CONFIGURATION FOR ADVANCED OCR & VALIDATION ---
class CONFIG:
    DET_ARCH = 'db_mobilenet_v3_large'
    RECO_ARCH = 'crnn_vgg16_bn'
    UPSCALE_FACTOR = 2.0
    NOTES_CROP_PERCENTAGE = 0.22
    BOX_THRESHOLD = 0.3
    WORD_CONF_THRESHOLD = 0.3
    TAG_CONFIDENCE_REVIEW_THRESHOLD = 0.8
    SYMBOL_CONFIDENCE_REVIEW_THRESHOLD = 0.5  # New: Flag ambiguous symbols
    GDRIVE_DATASET_PATH = Path("/content/drive/MyDrive/colab_data/pid_dataset")
    FUZZY_ACRONYM_MAP = {
        'GRO': 'GRP', 'DE1': 'DDL', 'ZLO': 'ZLC', '2L0': 'ZLC', 'GRI': 'GRP',
        'INS38C': 'INS(38C)', 'INS-52C': 'INS(52C)', 'ODL': 'DDL', 'SDI': 'SDL'
    }
    TEXT_CORRECTION_PATTERNS = {
        r'^([A-Z]{2})(\d{5})$': r'\1-\2', r'^(\d)([A-Z]{2})(\d{4})$': r'\1-\2-\3'
    }

# --- 2. LOAD AI MODELS & CLASS NAMES (GLOBAL STATE) ---
print("Loading AI Models... This may take a moment.")
try:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Could not find 'best.pt'. Make sure your trained model file is in the project folder.")
    symbol_detector = YOLO(MODEL_PATH)
    
    if CLASS_NAMES_PATH.is_file():
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
    else:
        class_names = list(symbol_detector.names.values())
        with open(CLASS_NAMES_PATH, 'w') as f:
            json.dump(class_names, f)
            
    text_detector = ocr_predictor(det_arch=CONFIG.DET_ARCH, reco_arch=CONFIG.RECO_ARCH, pretrained=True, export_as_straight_boxes=True)
    text_detector.det_predictor.model.postprocessor.box_thresh = CONFIG.BOX_THRESHOLD
    print("✅ All AI models loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Failed to load AI models: {e}")
    exit()

# --- 3. DATA FORMATTING FUNCTIONS ---
def format_to_xml(graph_data):
    root = ET.Element("PID_Graph")
    nodes_element = ET.SubElement(root, "Nodes")
    for node in graph_data:
        node_element = ET.SubElement(nodes_element, "Node", id=str(node["node_id"]))
        ET.SubElement(node_element, "Type").text = str(node["type"])
        ET.SubElement(node_element, "Tag").text = str(node.get("tag", "N/A"))
        ET.SubElement(node_element, "TagConfidence").text = str(node.get("tag_confidence", "N/A"))
        ET.SubElement(node_element, "NeedsReview").text = str(node.get("needs_review", False)).lower()
        ET.SubElement(node_element, "SymbolConfidence").text = str(node.get("symbol_confidence", "N/A"))
        ET.SubElement(node_element, "ConnectionValid").text = str(node.get("connection_valid", True)).lower()
        connections_element = ET.SubElement(node_element, "Connections")
        if "connections" in node and node["connections"]:
            ET.SubElement(connections_element, "ConnectedNode", id=str(node["connections"][0]))
    xml_str = ET.tostring(root, 'utf-8')
    return minidom.parseString(xml_str).toprettyxml(indent="  ")

def format_to_iso15926_json(graph_data):
    entities, relationships = [], []
    for node in graph_data:
        entities.append({
            "id": f"Node_{node['node_id']}",
            "class": "PhysicalObject",
            "attributes": {
                "type": node['type'],
                "tag": node.get('tag', 'N/A'),
                "tag_confidence": node.get('tag_confidence', 'N/A'),
                "symbol_confidence": node.get('symbol_confidence', 'N/A'),
                "needs_review": node.get('needs_review', False),
                "connection_valid": node.get('connection_valid', True)
            }
        })
        if "connections" in node and node["connections"]:
            if node['node_id'] < node["connections"][0]:
                relationships.append({
                    "from": f"Node_{node['node_id']}",
                    "to": f"Node_{node['connections'][0]}",
                    "type": "is-connected-to"
                })
    return {"entities": entities, "relationships": relationships}

# --- ADVANCED OCR HELPER FUNCTIONS ---
def auto_crop_main_diagram(img: np.ndarray):
    print("1. Auto-cropping the main diagram area...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_inv = cv2.bitwise_not(gray)
    kernel = np.ones((10, 10), np.uint8)
    dilated = cv2.dilate(gray_inv, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        padding = 30
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(img.shape[1] - x, w + 2 * padding), min(img.shape[0] - y, h + 2 * padding)
        cropped_img = img[y:y+h, x:x+w]
        print("✅ Cropping successful.")
        return cropped_img, x, y
    else:
        print("⚠️ Warning: Could not find a dominant contour. Using original image.")
        return img, 0, 0

def preprocess_image(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    if CONFIG.UPSCALE_FACTOR > 1.0:
        upscaled = cv2.resize(denoised, None, fx=CONFIG.UPSCALE_FACTOR, fy=CONFIG.UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    else:
        upscaled = denoised
    thresh = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4)
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh_inv = cv2.bitwise_not(closing)
    return cv2.cvtColor(thresh_inv, cv2.COLOR_GRAY2BGR)

def is_valid_text(text: str) -> bool:
    if not text: return False
    if text in CONFIG.FUZZY_ACRONYM_MAP.values() or text in ['STA', 'DDL', 'SDL', 'GRP']: return True
    if any(char.isalpha() for char in text) and any(char.isdigit() for char in text): return True
    if len(text) < 3: return False
    return True

def post_process_results(result):
    if not result.pages: return []
    words = [w for b in result.pages[0].blocks for l in b.lines for w in l.words]
    words = [{'value': re.sub(r'[^A-Z0-9\s-]', '', w.value), 'confidence': w.confidence, 'geometry': w.geometry} for w in words]
    print("\nDEBUG: Raw OCR words before merging:")
    for word in words:
        print(f"- Text: '{word['value']}', Confidence: {word['confidence']:.2f}, Geometry: {word['geometry']}")
    while True:
        merged_in_this_pass = False
        next_pass_words, used_indices = [], set()
        words.sort(key=lambda w: (w['geometry'][0][1], w['geometry'][0][0]))
        for i in range(len(words)):
            if i in used_indices: continue
            current_word, best_merge_candidate, min_dist = words[i], None, 0.03
            for j in range(i + 1, len(words)):
                if j in used_indices: continue
                candidate_word = words[j]
                y_center_current = (current_word['geometry'][0][1] + current_word['geometry'][1][1]) / 2
                y_center_candidate = (candidate_word['geometry'][0][1] + candidate_word['geometry'][1][1]) / 2
                if abs(y_center_current - y_center_candidate) < 0.02:
                    dist = candidate_word['geometry'][0][0] - current_word['geometry'][1][0]
                    if 0 <= dist < min_dist and not current_word['value'].startswith('-') and candidate_word['value'].startswith('-'):
                        best_merge_candidate = candidate_word
                        used_indices.add(j)
                        break
            if best_merge_candidate:
                new_value = current_word['value'] + best_merge_candidate['value']
                new_confidence = (current_word['confidence'] + best_merge_candidate['confidence']) / 2
                new_geometry = (current_word['geometry'][0], best_merge_candidate['geometry'][1])
                next_pass_words.append({'value': new_value, 'confidence': new_confidence, 'geometry': new_geometry})
                used_indices.add(i); merged_in_this_pass = True
            else:
                next_pass_words.append(current_word)
        words = next_pass_words
        if not merged_in_this_pass: break
    final_results = []
    for word in words:
        text = word['value']
        if text in CONFIG.FUZZY_ACRONYM_MAP: text = CONFIG.FUZZY_ACRONYM_MAP[text]
        for pattern, replacement in CONFIG.TEXT_CORRECTION_PATTERNS.items():
            text = re.sub(pattern, replacement, text)
        if is_valid_text(text):
            word['value'] = text
            final_results.append(word)
        else:
            print(f"DEBUG: Filtered out text: '{text}', Confidence: {word['confidence']:.2f}")
    return final_results

# --- 4. THE MAIN AI PIPELINE ---
def digitize_pid_image(original_img):
    print("\n--- Starting Enhanced Digitization Pipeline ---")
    
    # --- STAGE A: SYMBOL DETECTION ---
    results_yolo = symbol_detector(original_img, verbose=False)[0]
    yolo_data = []
    for box in results_yolo.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf.item()
        class_name = symbol_detector.names[int(box.cls.item())]
        if conf > 0.4:
            yolo_data.append({"bbox": (x1, y1, x2, y2), "class_name": class_name, "conf": conf})
    print(f"✅ Detected {len(yolo_data)} high-confidence symbols.")
    
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    
    # --- STAGE B: TEXT DETECTION (ADVANCED OCR) ---
    main_diagram_cropped, crop_offset_x, crop_offset_y = auto_crop_main_diagram(original_img)
    print("1b. Cropping out the notes area...")
    crop_width_px = int(main_diagram_cropped.shape[1] * (1 - CONFIG.NOTES_CROP_PERCENTAGE))
    img_without_notes = main_diagram_cropped[:, :crop_width_px]
    print("✅ Notes cropped out.")

    processed_img = preprocess_image(img_without_notes)
    processed_h, processed_w = processed_img.shape[:2]

    print("\n4. Running Doctr OCR...")
    result = text_detector([processed_img])

    final_results = post_process_results(result)

    ocr_results_list = []
    for item in final_results:
        geo = item['geometry']
        x1 = int(geo[0][0] * processed_w / CONFIG.UPSCALE_FACTOR) + crop_offset_x
        y1 = int(geo[0][1] * processed_h / CONFIG.UPSCALE_FACTOR) + crop_offset_y
        x2 = int(geo[1][0] * processed_w / CONFIG.UPSCALE_FACTOR) + crop_offset_x
        y2 = int(geo[1][1] * processed_h / CONFIG.UPSCALE_FACTOR) + crop_offset_y
        text = item['value']
        conf = item['confidence']
        ocr_results_list.append({"text": text, "bbox": (x1, y1, x2, y2), "confidence": conf})
    print(f"✅ Recognized {len(ocr_results_list)} text blocks.")

    print("\nDEBUG: All detected text blocks:")
    for ocr_res in ocr_results_list:
        print(f"- Text: '{ocr_res['text']}', Confidence: {ocr_res['confidence']:.2f}, BBox: {ocr_res['bbox']}")
    
    # --- STAGE C: LINE AND ARROW DETECTION ---
    mask = np.ones_like(gray_img) * 255
    for symbol in yolo_data:
        x1, y1, x2, y2 = symbol["bbox"]
        mask[y1:y2, x1:x2] = 0
    for ocr_res in ocr_results_list:
        x1, y1, x2, y2 = ocr_res["bbox"]
        mask[y1:y2, x1:x2] = 0
    gray_img = cv2.bitwise_and(gray_img, mask)
    
    _, binary_full = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton_full = skeletonize(binary_full // 255).astype(np.uint8) * 255
    
    scale_factor = 0.5
    img_small = cv2.resize(gray_img, None, fx=scale_factor, fy=scale_factor)
    _, binary = cv2.threshold(img_small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    skeleton = skeletonize(binary // 255).astype(np.uint8) * 255
    edges = cv2.Canny(skeleton, 75, 200)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=75, maxLineGap=20)
    
    def preprocess_lines(lines, extend_by=15):
        if lines is None:
            return []
        processed_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx, dy = dx/length, dy/length
                x1, y1 = int(x1 - dx * extend_by), int(y1 - dy * extend_by)
                x2, y2 = int(x2 + dx * extend_by), int(y2 + dy * extend_by)
            processed_lines.append((x1, y1, x2, y2))
    
        horizontal = sorted([l for l in processed_lines if abs(l[2]-l[0]) > abs(l[3]-l[1])], key=lambda l: (l[1], l[0]))
        vertical = sorted([l for l in processed_lines if abs(l[2]-l[0]) <= abs(l[3]-l[1])], key=lambda l: (l[0], l[1]))
        processed_lines = horizontal + vertical
    
        def are_collinear(line1, line2, angle_tol=3, dist_tol=15):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            v1 = np.array([x2 - x1, y2 - y1])
            v2 = np.array([x4 - x3, y4 - y3])
            len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if len1 == 0 or len2 == 0:
                return False
            cos_angle = np.dot(v1, v2) / (len1 * len2)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            if angle > 180:
                angle = 360 - angle
            if angle > angle_tol:
                return False
            line1_shapely = LineString([(x1, y1), (x2, y2)])
            return min(line1_shapely.distance(Point(p)) for p in [(x3, y3), (x4, y4)]) < dist_tol
    
        merged_lines = []
        while processed_lines:
            current = processed_lines.pop(0)
            i = 0
            while i < len(processed_lines):
                if are_collinear(current, processed_lines[i]):
                    x1, y1, x2, y2 = current
                    x3, y3, x4, y4 = processed_lines.pop(i)
                    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                    xs, ys = zip(*points)
                    current = (min(xs), min(ys), max(xs), max(ys)) if abs(max(xs)-min(xs)) > abs(max(ys)-min(ys)) else (min(xs), min(ys), max(xs), max(ys))
                else:
                    i += 1
            merged_lines.append(current)
        return [(int(x1/scale_factor), int(y1/scale_factor), int(x2/scale_factor), int(y2/scale_factor)) for x1, y1, x2, y2 in merged_lines]
    
    line_data = preprocess_lines(lines)
    print(f"✅ Detected {len(line_data)} processed line segments.")
    
    def classify_lines(lines, gap_threshold=25, min_segments_dotted=2):
        solid_lines = []
        dotted_lines = []
        horizontal = []
        vertical = []
        for line in lines:
            x1, y1, x2, y2 = line
            if abs(x2 - x1) > abs(y2 - y1):
                horizontal.append(line)
            else:
                vertical.append(line)
    
        def cluster_lines(line_group, gap_threshold):
            clusters = []
            current_cluster = [line_group[0]] if line_group else []
            for line in line_group[1:]:
                x1, y1, x2, y2 = line
                last_x1, last_y1, last_x2, last_y2 = current_cluster[-1]
                dist = min(np.sqrt((x1 - last_x2)**2 + (y1 - last_y2)**2),
                           np.sqrt((x2 - last_x1)**2 + (y2 - last_y1)**2))
                if dist < gap_threshold:
                    current_cluster.append(line)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [line]
            if current_cluster:
                clusters.append(current_cluster)
            return clusters
    
        for line_group in [horizontal, vertical]:
            clusters = cluster_lines(line_group, gap_threshold)
            for cluster in clusters:
                if len(cluster) >= min_segments_dotted:
                    gaps = []
                    for i in range(1, len(cluster)):
                        x1, y1, x2, y2 = cluster[i-1]
                        nx1, ny1, nx2, ny2 = cluster[i]
                        gap = min(np.sqrt((nx1 - x2)**2 + (ny1 - y2)**2),
                                  np.sqrt((nx2 - x1)**2 + (ny2 - y1)**2))
                        gaps.append(gap)
                    if gaps and max(gaps) < gap_threshold:
                        dotted_lines.extend(cluster)
                    else:
                        solid_lines.extend(cluster)
                else:
                    solid_lines.extend(cluster)
    
        return solid_lines, dotted_lines
    
    solid_lines, dotted_lines = classify_lines(line_data)
    print(f"✅ Classified {len(solid_lines)} solid lines and {len(dotted_lines)} dotted lines.")
    
    print(f"✅ Detected 0 arrows.")
    
    # --- STAGE D: GRAPH CONSTRUCTION ---
    G = nx.Graph()
    bom_counts = Counter()
    
    node_id_counter = 1
    symbol_nodes = {}
    for symbol in yolo_data:
        node_id = node_id_counter
        x1, y1, x2, y2 = symbol["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best_tag_data = min(
            [(ocr_res, np.sqrt(((x1+x2)/2 - (ocr_res["bbox"][0]+ocr_res["bbox"][2])/2)**2 + ((y1+y2)/2 - (ocr_res["bbox"][1]+ocr_res["bbox"][3])/2)**2))
             for ocr_res in ocr_results_list],
            key=lambda x: x[1],
            default=(None, float('inf'))
        )
        best_tag, best_tag_conf = (best_tag_data[0]["text"], best_tag_data[0]["confidence"]) if best_tag_data[0] and best_tag_data[1] < 300 else (None, 0.0)
        needs_review = best_tag_conf < CONFIG.TAG_CONFIDENCE_REVIEW_THRESHOLD or best_tag is None
        symbol_confidence = symbol["conf"]
        if symbol_confidence < CONFIG.SYMBOL_CONFIDENCE_REVIEW_THRESHOLD:
            needs_review = True  # Flag ambiguous symbols
        attrs = {
            "type": symbol["class_name"],
            "tag": best_tag,
            "tag_confidence": best_tag_conf,
            "symbol_confidence": symbol_confidence,
            "needs_review": needs_review,
            "coords": (cx, cy),
            "bbox": symbol["bbox"]
        }
        G.add_node(node_id, **attrs)
        symbol_nodes[node_id] = attrs
        bom_counts[symbol["class_name"]] += 1
        node_id_counter += 1
    
    def find_junctions(lines, threshold=75, min_angle=45):
        junctions = set()
        for i, line1 in enumerate(lines):
            line1_shapely = LineString([line1[:2], line1[2:]])
            for line2 in lines[i+1:]:
                line2_shapely = LineString([line2[:2], line2[2:]])
                if line1_shapely.intersects(line2_shapely):
                    inter = line1_shapely.intersection(line2_shapely)
                    if inter.geom_type == 'Point':
                        x1, y1, x2, y2 = line1
                        x3, y3, x4, y4 = line2
                        v1 = np.array([x2 - x1, y2 - y1])
                        v2 = np.array([x4 - x3, y4 - y3])
                        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
                        if angle < min_angle or angle > 180 - min_angle:
                            continue
                        junctions.add((inter.x, inter.y))
        return list(junctions)
    
    junctions = find_junctions(line_data)
    print(f"✅ Detected {len(junctions)} junction points.")
    
    def is_line_near_symbol(line, bbox, threshold=75):
        line_shapely = LineString([line[:2], line[2:]])
        symbol_points = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3])]
        return any(line_shapely.distance(Point(p)) < threshold for p in symbol_points)
    
    def get_nearest_neighbor(node_id, G, junctions):
        min_dist = float('inf')
        nearest_node = None
        node_coords = G.nodes[node_id]["coords"]
        for other_id in G.nodes:
            if other_id != node_id:
                other_coords = G.nodes[other_id]["coords"]
                dist = np.sqrt((node_coords[0] - other_coords[0])**2 + (node_coords[1] - other_coords[1])**2)
                for jx, jy in junctions:
                    if (LineString([node_coords, (jx, jy)]).distance(Point(other_coords)) < 20):
                        if dist < min_dist:
                            min_dist = dist
                            nearest_node = other_id
        return nearest_node
    
    for node_id in G.nodes:
        connected_node = get_nearest_neighbor(node_id, G, junctions)
        if connected_node and not G.has_edge(node_id, connected_node):
            G.add_edge(node_id, connected_node, direction="unknown")
            
    components = list(nx.connected_components(G))
    if len(components) > 1:
        for i in range(len(components) - 1):
            for j in range(i + 1, len(components)):
                comp_i, comp_j = components[i], components[j]
                if comp_i and comp_j:
                    node_i = min(comp_i)
                    node_j = min(comp_j)
                    if not G.has_edge(node_i, node_j):
                        G.add_edge(node_i, node_j, direction="unknown")
    
    def propagate_flow(G):
        potential_sources = [n for n in G if G.nodes[n]["type"] in ["motor", "silencer"]]
        if not potential_sources:
            potential_sources = [min(G.nodes)]
        for start in potential_sources:
            queue = deque([(start, "downstream")])
            visited = set()
            while queue:
                node, direction = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        G.edges[node, neighbor]["direction"] = direction
                        queue.append((neighbor, direction))
    
    propagate_flow(G)
    
    # --- Validate Connections and Build Graph Data ---
    node_ids = set(G.nodes)
    missing_tags = 0
    broken_connections = 0
    ambiguous_symbols = 0
    final_img = original_img.copy()
    for line in solid_lines:
        cv2.line(final_img, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
    for line in dotted_lines:
        cv2.line(final_img, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)
    graph_data_json = []
    for node_id in G.nodes:
        data = G.nodes[node_id]
        x1, y1, x2, y2 = data["bbox"]
        cv2.rectangle(final_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        tag_text = f"{data['tag'] or 'NO_TAG'}[{data['tag_confidence']:.2f}]" if data['tag'] else "NO_TAG"
        cv2.putText(final_img, f"{node_id}:{tag_text}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        connections = [n for n in G.neighbors(node_id)]
        # Validate connections
        valid_connections = [n for n in connections if n in node_ids]
        connection_valid = len(valid_connections) == len(connections)
        if not connection_valid:
            broken_connections += 1
        if data["tag"] is None:
            missing_tags += 1
        if data["symbol_confidence"] < CONFIG.SYMBOL_CONFIDENCE_REVIEW_THRESHOLD:
            ambiguous_symbols += 1
        node_data = {
            "node_id": node_id,
            "type": data['type'],
            "tag": data['tag'],
            "tag_confidence": data['tag_confidence'],
            "symbol_confidence": data['symbol_confidence'],
            "needs_review": data['needs_review'],
            "connections": [valid_connections[0]] if valid_connections else [],
            "connection_valid": connection_valid,
            "bbox": data["bbox"]
        }
        graph_data_json.append(node_data)
    for u, v in G.edges:
        if u < v:
            start_point = (int(G.nodes[u]["coords"][0]), int(G.nodes[u]["coords"][1]))
            end_point = (int(G.nodes[v]["coords"][0]), int(G.nodes[v]["coords"][1]))
            dir = G.edges[u, v].get("direction", "unknown")
            if dir == "downstream":
                cv2.arrowedLine(final_img, start_point, end_point, (0, 255, 0), 3)
            elif dir == "upstream":
                cv2.arrowedLine(final_img, end_point, start_point, (0, 255, 0), 3)
            else:
                cv2.line(final_img, start_point, end_point, (0, 255, 0), 3)
    
    review_summary = {
        "missing_tags": missing_tags,
        "broken_connections": broken_connections,
        "ambiguous_symbols": ambiguous_symbols
    }
    return graph_data_json, final_img, skeleton_full, bom_counts, ocr_results_list, review_summary

# --- 5. DEFINE THE WEB SERVER ENDPOINTS ---
@app.route('/digitize', methods=['POST'])
def digitize():
    print("Received request for /digitize")
    try:
        if 'file' not in request.files:
            print("❌ No file part in request")
            return jsonify({"error": "No file part in request"}), 400
        file = request.files['file']
        if file.filename == '':
            print("❌ No selected file")
            return jsonify({"error": "No selected file"}), 400
        
        file_bytes = file.read()
        print(f"Received file: {file.filename}, size: {len(file_bytes)} bytes")
        poppler_path = os.getenv('POPPLER_PATH', None)
        if file.filename.lower().endswith('.pdf'):
            try:
                images = convert_from_bytes(file_bytes, first_page=1, last_page=1, poppler_path=poppler_path)
                if not images:
                    print("❌ Failed to process PDF: No images extracted")
                    raise ValueError("Could not process PDF: No images extracted")
                original_img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"❌ PDF processing error: {str(e)}")
                return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 400
        else:
            original_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if original_img is None:
                print("❌ Failed to decode image")
                return jsonify({"error": "Failed to decode image: Invalid or corrupted file"}), 400

        graph_data_json, final_img, skeleton, bom_counts, ocr_results_list, review_summary = digitize_pid_image(original_img)
        
        final_img[skeleton == 1] = [200, 200, 200]
        
        _, buffer = cv2.imencode('.jpg', final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print("✅ Successfully digitized image. Sending response.")
        return jsonify({
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "graph_data": {
                "json": graph_data_json,
                "xml": format_to_xml(graph_data_json),
                "iso15926_json": format_to_iso15926_json(graph_data_json)
            },
            "bill_of_materials": [{"symbol": name, "count": count} for name, count in bom_counts.items()],
            "class_names": class_names,
            "ocr_results": [{"text": res["text"], "bbox": res["bbox"], "confidence": res["confidence"]} for res in ocr_results_list],
            "review_summary": review_summary
        })
    except Exception as e:
        print(f"❌ An error occurred during digitization: {e}")
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {str(e)}. Check the backend terminal for details."}), 500

@app.route('/submit_correction', methods=['POST'])
def submit_correction():
    global class_names
    try:
        data = request.get_json()
        annotation = data['annotation']
        class_name = annotation['className'].strip()
        is_new_class = class_name not in class_names
        
        if is_new_class:
            return jsonify({"status": "new_class_detected", "className": class_name})

        image_b64 = data['image_b64'].split(',')[1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
        h, w, _ = img.shape
        timestamp = int(time.time())
        img_filename = f"correction_{timestamp}.jpg"
        label_filename = f"correction_{timestamp}.txt"
        cv2.imwrite(str(FEEDBACK_IMAGES_DIR / img_filename), img)
        class_id = class_names.index(class_name)
        x1, y1, x2, y2 = annotation['box']
        cx, cy, box_w, box_h = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
        with open(FEEDBACK_LABELS_DIR / label_filename, 'w') as f: f.write(f"{class_id} {cx} {cy} {box_w} {box_h}")
        
        return jsonify({"status": "success", "message": "Correction saved."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to save correction: {str(e)}"}), 500

@app.route('/submit_tag_correction', methods=['POST'])
def submit_tag_correction():
    try:
        data = request.get_json()
        node_id = data.get('node_id')
        tag = data.get('tag')
        image_b64 = data.get('image_b64', '').split(',')[1] if data.get('image_b64') else None
        
        if not node_id or not tag:
            print(f"❌ Missing node_id or tag: node_id={node_id}, tag={tag}")
            return jsonify({"error": "Missing node_id or tag"}), 400
        
        timestamp = int(time.time())
        tag_filename = f"tag_correction_{timestamp}_{node_id}.json"
        correction_data = {
            "node_id": node_id,
            "tag": tag,
            "timestamp": timestamp,
            "tag_confidence": 1.0,
            "needs_review": False
        }
        
        if image_b64:
            try:
                img = cv2.imdecode(np.frombuffer(base64.b64decode(image_b64), np.uint8), cv2.IMREAD_COLOR)
                img_filename = f"tag_correction_{timestamp}_{node_id}.jpg"
                cv2.imwrite(str(FEEDBACK_IMAGES_DIR / img_filename), img)
                correction_data["image_filename"] = img_filename
            except Exception as e:
                print(f"⚠️ Warning: Failed to save tag correction image: {str(e)}")
        
        with open(FEEDBACK_TAGS_DIR / tag_filename, 'w') as f:
            json.dump(correction_data, f)
        
        print(f"✅ Saved tag correction for node {node_id}: {tag}")
        return jsonify({
            "status": "success",
            "message": f"Tag correction saved for node {node_id}.",
            "updated_node": {
                "node_id": node_id,
                "tag": tag,
                "tag_confidence": 1.0,
                "needs_review": False
            }
        })
    except Exception as e:
        print(f"❌ Failed to save tag correction: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to save tag correction: {str(e)}"}), 500

@app.route('/upload_new_symbol_data', methods=['POST'])
def upload_new_symbol_data():
    global class_names
    try:
        class_name = request.form['className']
        files = request.files.getlist('files')
        
        if class_name not in class_names:
            class_names.append(class_name)
            with open(CLASS_NAMES_PATH, 'w') as f:
                json.dump(class_names, f)
        
        class_id = class_names.index(class_name)
        
        for file in files:
            timestamp = int(time.time() * 1000)
            img_bytes = file.read()
            img_np = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            img_filename = f"new_class_{timestamp}_{Path(file.filename).stem}.jpg"
            label_filename = f"new_class_{timestamp}_{Path(file.filename).stem}.txt"
            cv2.imwrite(str(FEEDBACK_IMAGES_DIR / img_filename), img)
            
            yolo_label = f"{class_id} 0.5 0.5 1.0 1.0"
            with open(FEEDBACK_LABELS_DIR / label_filename, 'w') as f: f.write(yolo_label)
            
        return jsonify({"status": "success", "message": f"Added {len(files)} new files for '{class_name}'."})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Failed to process new symbol data: {str(e)}"}), 500

@app.route('/start_finetuning', methods=['POST'])
def start_finetuning():
    try:
        python_executable = sys.executable
        print(f"🚀 Starting fine-tuning process with {python_executable} finetune.py")
        subprocess.Popen([python_executable, "finetune.py"], cwd=os.getcwd())
        return jsonify({"status": "success", "message": "Fine-tuning started in the background."})
    except Exception as e:
        print(f"❌ Failed to start fine-tuning: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to start fine-tuning: {str(e)}"}), 500

def watch_for_restart():
    while True:
        if RESTART_FLAG_FILE.exists():
            print("🚨 New model detected! Restarting server...")
            RESTART_FLAG_FILE.unlink()
            os.execv(sys.executable, ['python'] + sys.argv)
        time.sleep(10)

if __name__ == '__main__':
    watcher = threading.Thread(target=watch_for_restart, daemon=True)
    watcher.start()
    print("✅ Backend server started on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000)
