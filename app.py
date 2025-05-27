from flask import Flask, Response, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import subprocess
import logging
import time
import signal
import os
import threading
from queue import Queue
from logging.handlers import RotatingFileHandler
import json

app = Flask(__name__)

# Configuration du logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler = RotatingFileHandler('app.log', maxBytes=1e6, backupCount=3)
log_handler.setFormatter(log_formatter)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)
logging.basicConfig(level=logging.INFO)

# Paramètres configurables
CONFIG = {
    "object": {"width": 20, "height": 20},  # Dimensions de référence en cm
    "focal_length": None,
    "detection": {
        "enabled": False,
        "confidence": 0.4,
        "frame_skip": 3  # Traiter 1 frame sur N
    },
    "camera": {
        "width": 640,
        "height": 480,
        "framerate": 15,
        "brightness": 0.2,
        "contrast": 1.2,
        "flip": True
    }
}

# Variables globales
model = None
camera_process = None
frame_queue = Queue(maxsize=2)
last_detected_objects = []
detection_active = False

def init_model():
    """Initialise le modèle YOLO"""
    global model
    try:
        model = YOLO("yolov8n.pt").to('cpu')
        model.fuse()
        logger.info("Modèle YOLO chargé avec succès")
        return True
    except Exception as e:
        logger.error(f"Erreur chargement modèle YOLO: {e}")
        return False

def cleanup_resources():
    """Nettoyage des ressources"""
    global camera_process, model
    if camera_process:
        camera_process.terminate()
        camera_process.wait()
        camera_process = None
    model = None

def get_camera_command():
    """Configuration de la caméra"""
    cmd = [
        "libcamera-vid",
        "-t", "0",
        "--width", str(CONFIG["camera"]["width"]),
        "--height", str(CONFIG["camera"]["height"]),
        "--framerate", str(CONFIG["camera"]["framerate"]),
        "--codec", "mjpeg",
        "-o", "-",
        "--nopreview",
        "--brightness", str(CONFIG["camera"]["brightness"]),
        "--contrast", str(CONFIG["camera"]["contrast"]),
        "--denoise", "cdn_off",
        "--sharpness", "0",
        "--saturation", "1.0"
    ]
    
    if CONFIG["camera"]["flip"]:
        cmd.extend(["--vflip", "--hflip"])
    
    return cmd

def calculate_distance(pixel_width):
    """Calcul de distance"""
    if CONFIG["focal_length"] is None:
        return None
    return (CONFIG["object"]["width"] * CONFIG["focal_length"]) / pixel_width

def camera_capture_thread():
    """Thread de capture vidéo"""
    global camera_process
    
    while True:
        try:
            cleanup_resources()
            process = subprocess.Popen(
                get_camera_command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            camera_process = process
            
            buffer = b""
            
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    if process.poll() is not None:
                        break
                    time.sleep(0.1)
                    continue
                
                buffer += chunk
                jpg_start = buffer.find(b"\xff\xd8")
                jpg_end = buffer.find(b"\xff\xd9")
                
                if jpg_start != -1 and jpg_end != -1:
                    frame_data = buffer[jpg_start:jpg_end+2]
                    buffer = buffer[jpg_end+2:]
                    
                    if frame_queue.full():
                        frame_queue.get()
                    frame_queue.put(frame_data)
        
        except Exception as e:
            logger.error(f"Erreur capture vidéo: {e}")
            time.sleep(1)

def process_frame(frame_data):
    """Traitement d'une frame"""
    global last_detected_objects
    
    img = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    frame_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    current_objects = []
    
    if detection_active and model is not None:
        results = model(frame_np, 
                       imgsz=320, 
                       conf=CONFIG["detection"]["confidence"], 
                       device='cpu')
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            if label == "person":
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width_px = x2 - x1
            height_px = y2 - y1
            
            if CONFIG["focal_length"] is None and 20 < width_px < 500:
                CONFIG["focal_length"] = (width_px * 50) / CONFIG["object"]["width"]
                logger.info(f"Calibration: focal_length={CONFIG['focal_length']:.1f}")
            
            distance = calculate_distance(width_px)
            
            current_objects.append({
                "label": label,
                "distance": distance,
                "width_px": width_px,
                "height_px": height_px,
                "confidence": float(box.conf[0])
            })
            
            color = (0, 255, 0)
            cv2.rectangle(frame_np, (x1, y1), (x2, y2), color, 2)
            
            info_lines = [
                f"Objet: {label}",
                f"Distance: {distance:.1f}cm" if distance else "Distance: N/A",
                f"Dimensions: {width_px}x{height_px} px"
            ]
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            for i, line in enumerate(info_lines):
                y_pos = y1 - 10 - (i * 20)
                cv2.putText(frame_np, line, (x1, max(y_pos, 15)), 
                            font, font_scale, (255, 255, 255), thickness)
    
    last_detected_objects = current_objects
    _, jpeg = cv2.imencode('.jpg', cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR), 
                          [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return jpeg.tobytes()

def processing_thread():
    """Thread de traitement"""
    global detection_active
    frame_counter = 0
    
    while True:
        try:
            if frame_queue.empty():
                time.sleep(0.01)
                continue
                
            frame_data = frame_queue.get()
            frame_counter += 1
            
            if not detection_active or frame_counter % CONFIG["detection"]["frame_skip"] != 0:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                continue
                
            processed_frame = process_frame(frame_data)
            if processed_frame:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + processed_frame + b'\r\n')
        
        except Exception as e:
            logger.error(f"Erreur traitement frame: {e}")
            time.sleep(0.1)

def gen_frames():
    return processing_thread()

@app.route('/')
def index():
    return render_template('index.html',
                         detection_status=detection_active,
                         confidence=CONFIG["detection"]["confidence"],
                         frame_skip=CONFIG["detection"]["frame_skip"])

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_data')
def detection_data():
    def generate():
        while True:
            try:
                yield f"data: {json.dumps(last_detected_objects)}\n\n"
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Erreur SSE: {e}")
                time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

@app.route('/toggle_detection')
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    
    if detection_active and model is None:
        if not init_model():
            detection_active = False
            return "Erreur: Impossible de charger le modèle", 500
    
    status = "activée" if detection_active else "désactivée"
    logger.info(f"Détection {status}")
    return jsonify({"status": status, "success": True})

@app.route('/update_settings', methods=['POST'])
def update_settings():
    try:
        CONFIG["detection"]["confidence"] = float(request.form.get('confidence', 0.4))
        CONFIG["detection"]["frame_skip"] = int(request.form.get('frame_skip', 3))
        logger.info(f"Paramètres mis à jour - Confiance: {CONFIG['detection']['confidence']}, Skip: {CONFIG['detection']['frame_skip']}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Erreur mise à jour paramètres: {e}")
        return jsonify({"success": False}), 400

@app.route('/recalibrate')
def recalibrate():
    CONFIG["focal_length"] = None
    logger.info("Recalibration déclenchée")
    return "Recalibration en cours..."

def handle_shutdown(signum, frame):
    logger.info("Arrêt propre du système...")
    cleanup_resources()
    os._exit(0)

if __name__ == '__main__':
    # Configurer les handlers de signal
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    # Démarrer les threads
    threading.Thread(target=camera_capture_thread, daemon=True).start()
    
    # Démarrer Flask
    app.run(host='0.0.0.0', port=5050, threaded=True, debug=False)
