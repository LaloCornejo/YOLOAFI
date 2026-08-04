from ultralytics import YOLO
import cv2
import subprocess
import json
from datetime import datetime

model = YOLO('COCO.pt') 

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30

ffplay = subprocess.Popen([
    'ffplay', '-f', 'rawvideo', '-pixel_format', 'bgr24',
    '-video_size', f'{width}x{height}',
    '-framerate', str(fps),
    '-i', '-'
], stdin=subprocess.PIPE)

allowed_classes = {'person', 'bottle', 'cell phone', 'laptop', 'backpack', 'handbag', 'cup', 'bag'}

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

tracked_people = {}
next_person_id = 0
frame_count = 0
IOU_THRESHOLD = 0.3
CONFIRMATION_FRAMES = 10
MAX_MISSING_FRAMES = 30

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        results = model.predict(frame, device='cuda:0', verbose=False, conf=0.65)
        annotated_frame = frame.copy()
        
        current_detections = []
        for box in results[0].boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if class_name == 'person' and confidence > 0.50:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                current_detections.append((x1, y1, x2, y2, confidence))
        
        matched_ids = set()
        for detection in current_detections:
            x1, y1, x2, y2, conf = detection
            best_match_id = None
            best_iou = IOU_THRESHOLD
            
            for person_id, person_data in tracked_people.items():
                if person_data['missing_frames'] < MAX_MISSING_FRAMES:
                    iou = calculate_iou((x1, y1, x2, y2), person_data['last_box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_match_id = person_id
            
            if best_match_id is not None:
                tracked_people[best_match_id]['last_box'] = (x1, y1, x2, y2)
                tracked_people[best_match_id]['missing_frames'] = 0
                tracked_people[best_match_id]['consecutive_frames'] += 1
                matched_ids.add(best_match_id)
                
                if tracked_people[best_match_id]['consecutive_frames'] >= CONFIRMATION_FRAMES:
                    tracked_people[best_match_id]['confirmed'] = True
            else:
                tracked_people[next_person_id] = {
                    'last_box': (x1, y1, x2, y2),
                    'first_seen': datetime.now().isoformat(),
                    'consecutive_frames': 1,
                    'missing_frames': 0,
                    'confirmed': False
                }
                matched_ids.add(next_person_id)
                next_person_id += 1
        
        for person_id in list(tracked_people.keys()):
            if person_id not in matched_ids:
                tracked_people[person_id]['missing_frames'] += 1
                if tracked_people[person_id]['missing_frames'] >= MAX_MISSING_FRAMES:
                    del tracked_people[person_id]
        
        for person_id, person_data in tracked_people.items():
            if person_data['confirmed'] and person_data['missing_frames'] == 0:
                x1, y1, x2, y2 = person_data['last_box']
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'Person ID: {person_id}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for box in results[0].boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if class_name in allowed_classes and class_name != 'person' and confidence > 0.50:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f'{class_name}: {confidence:.2f}'
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        confirmed_count = sum(1 for p in tracked_people.values() if p['confirmed'])
        
        if frame_count % 30 == 0:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'confirmed_people_count': confirmed_count
            }
            with open('people_log.json', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        
        ffplay.stdin.write(annotated_frame.tobytes())
        
except (BrokenPipeError, KeyboardInterrupt):
    pass
finally:
    cap.release()
    ffplay.stdin.close()
    ffplay.wait()
