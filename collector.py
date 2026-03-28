"""
HUD Vision Suite  ·  Object Detection + Hand Gestures + Data Collector
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Three systems running in one camera feed:

  [1] Object Detector  — MediaPipe EfficientDet draws bounding boxes.
                          Each box is cropped → classified by your
                          Teachable Machine IMAGE model.

  [2] Hand Tracer      — MediaPipe Hands draws the full skeleton.
                          The hand crop → classified by your
                          Teachable Machine GESTURE model.
                          (You can use the same or a different TM model.)

  [3] Data Collector   — Hold a number key to continuously save
                          cropped snapshots into class folders,
                          ready to re-upload to Teachable Machine.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONE-TIME SETUP  (run this once to download the detector model):

    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/float32/latest/efficientdet_lite0.tflite",
        "efficientdet_lite0.tflite"
    )

Dependencies:
    pip install opencv-python mediapipe tensorflow numpy

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTROLS
    1  (hold)  →  save CAT crops
    2  (hold)  →  save PERSON crops
    3  (hold)  →  save HAND crops  (full hand bounding box)
    G          →  toggle glow
    D          →  toggle MediaPipe's own COCO label
    Q          →  quit
"""

import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import time
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ─────────────────────────────────────────────────────────────
#  ❶  PATHS  — edit these
# ─────────────────────────────────────────────────────────────

# Teachable Machine model used for OBJECT crops (cat, person, …)
OBJ_MODEL_PATH  = r"C:\Users\palaw\Documents\Projects\handsign\model\model.savedmodel"
OBJ_LABELS_PATH = r"C:\Users\palaw\Documents\Projects\handsign\model\labels.txt"

# Teachable Machine model used for HAND gesture crops
# If it's the same model, just point to the same paths.
HAND_MODEL_PATH  = r"C:\Users\palaw\Documents\Projects\handsign\model\model.savedmodel"
HAND_LABELS_PATH = r"C:\Users\palaw\Documents\Projects\handsign\model\labels.txt"

# MediaPipe EfficientDet tflite (downloaded once — see docstring)
MP_MODEL_PATH = "efficientdet_lite0.tflite"

# Where collected images are saved
COLLECT_ROOT = r"C:\Users\palaw\Documents\Projects\handsign\collected_data"

# ─────────────────────────────────────────────────────────────
#  ❷  DATA-COLLECTION CLASSES
#     Key 1 → classes[0], Key 2 → classes[1], Key 3 → classes[2]
#     Add / rename as you like.
# ─────────────────────────────────────────────────────────────
COLLECT_CLASSES = ["cat", "person", "hand"]

# ─────────────────────────────────────────────────────────────
#  ❸  TUNING
# ─────────────────────────────────────────────────────────────
IMG_SIZE         = 224
SMOOTHING_QUEUE  = 5
DET_THRESHOLD    = 0.40
MAX_OBJECTS      = 6
CAPTURE_INTERVAL = 0.12   # seconds between auto-saves while holding key

# ── Palette (BGR) ────────────────────────────────────────────
ACCENT      = (0, 230, 180)
ACCENT_DIM  = (0, 130, 100)
BOX_OBJ     = (0, 210, 255)   # cyan   — object boxes
BOX_HAND    = (80, 120, 255)  # orange — hand box
JOINT_CLR   = (255, 255, 255)
TIP_CLR     = (60, 200, 255)
FINGER_CLR  = (0, 210, 255)
LABEL_BG    = (18, 18, 24)
WHITE       = (240, 240, 240)
GRAY        = (100, 105, 115)
REC_CLR     = (50, 50, 220)   # red-ish record indicator

GLOW_ENABLED  = False
SHOW_MP_LABEL = False

# ─────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────
obj_model  = tf.saved_model.load(OBJ_MODEL_PATH)
obj_infer  = obj_model.signatures["serving_default"]

# Load hand model (may be the same object if paths are identical)
if HAND_MODEL_PATH == OBJ_MODEL_PATH:
    hand_infer = obj_infer
else:
    hand_model = tf.saved_model.load(HAND_MODEL_PATH)
    hand_infer = hand_model.signatures["serving_default"]

def _load_labels(path):
    with open(path) as f:
        return [" ".join(l.strip().split()[1:]) for l in f]

obj_labels  = _load_labels(OBJ_LABELS_PATH)
hand_labels = _load_labels(HAND_LABELS_PATH)

# Per-slot smoothing queues
obj_queues  = {}   # {slot_index: deque}
hand_queues = {}   # {hand_index: deque}

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE — Object Detector
# ─────────────────────────────────────────────────────────────
base_opts   = mp_python.BaseOptions(model_asset_path=MP_MODEL_PATH)
det_opts    = mp_vision.ObjectDetectorOptions(
    base_options=base_opts,
    score_threshold=DET_THRESHOLD,
    max_results=MAX_OBJECTS,
)
mp_detector = mp_vision.ObjectDetector.create_from_options(det_opts)

# ─────────────────────────────────────────────────────────────
#  MEDIAPIPE — Hands
# ─────────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
hands_det  = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
FINGERTIPS  = [4, 8, 12, 16, 20]
FINGER_NAMES = {4: "T", 8: "I", 12: "M", 16: "R", 20: "P"}

# ─────────────────────────────────────────────────────────────
#  DATA COLLECTION SETUP
# ─────────────────────────────────────────────────────────────
for cls in COLLECT_CLASSES:
    os.makedirs(os.path.join(COLLECT_ROOT, cls), exist_ok=True)

collect_counts    = {cls: len(os.listdir(os.path.join(COLLECT_ROOT, cls)))
                     for cls in COLLECT_CLASSES}
last_capture_time = 0.0


def save_crop(crop_bgr: np.ndarray, class_name: str) -> int:
    """Save crop to class folder. Returns new total count."""
    folder = os.path.join(COLLECT_ROOT, class_name)
    n      = len(os.listdir(folder))
    path   = os.path.join(folder, f"{class_name}_{n:05d}.jpg")
    cv2.imwrite(path, crop_bgr)
    return n + 1


# ─────────────────────────────────────────────────────────────
#  CLASSIFY HELPER
# ─────────────────────────────────────────────────────────────

def classify_crop(crop_bgr: np.ndarray, infer_fn, labels: list,
                  queue: deque) -> tuple[str, float]:
    """Resize crop → TM model → smoothed label + confidence."""
    img = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 255.0).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    preds = list(infer_fn(tf.constant(img)).values())[0].numpy()[0]
    queue.append(preds)
    avg   = np.mean(queue, axis=0)
    idx   = int(np.argmax(avg))
    return labels[idx], float(avg[idx])


# ─────────────────────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────────────────────

def blend(frame, overlay, alpha=0.55):
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx,cy), r, color, thickness)


def draw_corner_brackets(img, x1, y1, x2, y2, color, length=22, thickness=2):
    for tip, h, v in [
        ((x1,y1),(x1+length,y1),(x1,y1+length)),
        ((x2,y1),(x2-length,y1),(x2,y1+length)),
        ((x1,y2),(x1+length,y2),(x1,y2-length)),
        ((x2,y2),(x2-length,y2),(x2,y2-length)),
    ]:
        cv2.line(img, tip, h, color, thickness, cv2.LINE_AA)
        cv2.line(img, tip, v, color, thickness, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h, alpha=0.75):
    ov = frame.copy()
    draw_rounded_rect(ov, x, y, x+w, y+h, 10, LABEL_BG, -1)
    return blend(frame, ov, alpha)


def glow_line(frame, pt1, pt2, color, thickness=2):
    if GLOW_ENABLED:
        g = np.zeros_like(frame)
        cv2.line(g, pt1, pt2, color, thickness+6, cv2.LINE_AA)
        g = cv2.GaussianBlur(g, (0,0), sigmaX=3)
        cv2.addWeighted(g, 0.5, frame, 1.0, 0, frame)
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def glow_circle(frame, center, radius, color):
    if GLOW_ENABLED:
        g = np.zeros_like(frame)
        cv2.circle(g, center, radius+6, color, -1)
        g = cv2.GaussianBlur(g, (0,0), sigmaX=4)
        cv2.addWeighted(g, 0.45, frame, 1.0, 0, frame)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)


def glow_rect(frame, x1, y1, x2, y2, color, thickness=2):
    if GLOW_ENABLED:
        g = np.zeros_like(frame)
        cv2.rectangle(g, (x1,y1), (x2,y2), color, thickness+6, cv2.LINE_AA)
        g = cv2.GaussianBlur(g, (0,0), sigmaX=4)
        cv2.addWeighted(g, 0.5, frame, 1.0, 0, frame)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness, cv2.LINE_AA)


def draw_scan_line(frame, t):
    y  = int((t * 80) % frame.shape[0])
    ov = frame.copy()
    cv2.line(ov, (0,y), (frame.shape[1],y), ACCENT, 1)
    cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS {fps:.0f}",
                (frame.shape[1]-80, frame.shape[0]-14),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, GRAY, 1, cv2.LINE_AA)


def draw_conf_bar(frame, x1, y1, x2, y2, conf, color):
    cv2.rectangle(frame, (x1,y1), (x2,y2), (40,42,48), -1)
    fw = int(conf * (x2-x1))
    if fw > 0:
        cv2.rectangle(frame, (x1,y1), (x1+fw,y2), color, -1)


# ─────────────────────────────────────────────────────────────
#  DRAW OBJECT BOX
# ─────────────────────────────────────────────────────────────

def draw_object_box(frame, x1, y1, x2, y2,
                    tm_label, tm_conf,
                    mp_label, mp_conf,
                    obj_idx, is_collecting=False):
    H, W = frame.shape[:2]
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(W-1,x2), min(H-1,y2)

    box_color = REC_CLR if is_collecting else BOX_OBJ
    glow_rect(frame, x1, y1, x2, y2, box_color, thickness=2)
    draw_corner_brackets(frame, x1, y1, x2, y2, ACCENT, length=16, thickness=2)

    # index badge
    cv2.circle(frame, (x1+11, y1+11), 11, ACCENT, -1, cv2.LINE_AA)
    cv2.putText(frame, str(obj_idx+1),
                (x1+6, y1+16),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, LABEL_BG, 1, cv2.LINE_AA)

    # label panel
    extra_line = f"mp:{mp_label} {mp_conf*100:.0f}%" if SHOW_MP_LABEL else ""
    lines      = [tm_label.upper()] + ([extra_line] if extra_line else [])
    p_h = 20 + len(lines)*20
    p_w = max(160, max(len(l) for l in lines)*9 + 24)
    px  = x1
    py  = max(0, y1 - p_h - 6)
    frame = draw_panel(frame, px, py, p_w, p_h)

    cv2.putText(frame, lines[0],
                (px+8, py+18),
                cv2.FONT_HERSHEY_DUPLEX, 0.52, WHITE, 1, cv2.LINE_AA)
    if extra_line:
        cv2.putText(frame, extra_line,
                    (px+8, py+36),
                    cv2.FONT_HERSHEY_DUPLEX, 0.34, GRAY, 1, cv2.LINE_AA)

    # confidence bar
    bx1,by1 = px+8, py+p_h-8
    bx2,by2 = px+p_w-8, py+p_h-4
    draw_conf_bar(frame, bx1,by1,bx2,by2, tm_conf, ACCENT)
    cv2.putText(frame, f"{tm_conf*100:.1f}%",
                (bx2+4, by2+1),
                cv2.FONT_HERSHEY_DUPLEX, 0.34, ACCENT, 1, cv2.LINE_AA)

    # REC flash
    if is_collecting:
        cv2.putText(frame, "● REC",
                    (x1+4, y2-6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, REC_CLR, 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────
#  DRAW HAND SKELETON + GESTURE LABEL
# ─────────────────────────────────────────────────────────────

def draw_hand(frame, landmarks, W, H, hand_idx,
              gesture_label, gesture_conf,
              is_collecting=False):
    lm  = landmarks.landmark
    pts = [(int(lm[i].x*W), int(lm[i].y*H)) for i in range(21)]

    for a,b in CONNECTIONS:
        glow_line(frame, pts[a], pts[b], FINGER_CLR, thickness=2)

    for i, pt in enumerate(pts):
        if i in FINGERTIPS:
            glow_circle(frame, pt, 7, TIP_CLR)
            cv2.circle(frame, pt, 3, WHITE, -1, cv2.LINE_AA)
            cv2.putText(frame, FINGER_NAMES[i],
                        (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.32, TIP_CLR, 1, cv2.LINE_AA)
        elif i == 0:
            glow_circle(frame, pt, 6, ACCENT)
        else:
            cv2.circle(frame, pt, 3, JOINT_CLR, -1, cv2.LINE_AA)

    # hand bounding box
    xs,ys = [p[0] for p in pts],[p[1] for p in pts]
    pad   = 20
    bx1   = max(0, min(xs)-pad)
    by1   = max(0, min(ys)-pad)
    bx2   = min(W, max(xs)+pad)
    by2   = min(H, max(ys)+pad)

    box_color = REC_CLR if is_collecting else BOX_HAND
    draw_corner_brackets(frame, bx1, by1, bx2, by2, box_color, length=16, thickness=2)

    # gesture label panel (bottom of hand box)
    label_str = gesture_label.upper()
    p_w = max(160, len(label_str)*10 + 24)
    p_h = 48
    px  = bx1
    py  = min(H - p_h - 4, by2 + 6)
    frame = draw_panel(frame, px, py, p_w, p_h)

    cv2.putText(frame, "GESTURE",
                (px+8, py+14),
                cv2.FONT_HERSHEY_DUPLEX, 0.34, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, label_str,
                (px+8, py+34),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, WHITE, 1, cv2.LINE_AA)

    draw_conf_bar(frame, px+8, py+40, px+p_w-8, py+44, gesture_conf, ACCENT)

    if is_collecting:
        cv2.putText(frame, "● REC",
                    (bx1+4, by1-6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38, REC_CLR, 1, cv2.LINE_AA)

    return frame, (bx1, by1, bx2, by2)


# ─────────────────────────────────────────────────────────────
#  DATA COLLECTION HUD  (right-side panel)
# ─────────────────────────────────────────────────────────────

def draw_collector_panel(frame, active_key: int | None, now: float):
    H, W = frame.shape[:2]
    p_w  = 200
    rows = len(COLLECT_CLASSES)
    p_h  = 28 + rows * 32 + 10
    px   = W - p_w - 16
    py   = H - p_h - 16

    frame = draw_panel(frame, px, py, p_w, p_h, alpha=0.78)
    cv2.line(frame, (px+10, py), (px+p_w-10, py), ACCENT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "DATA COLLECTOR",
                (px+10, py+16),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, ACCENT, 1, cv2.LINE_AA)

    for i, cls in enumerate(COLLECT_CLASSES):
        ry       = py + 28 + i*32
        is_active = (active_key == i+1)
        key_str  = f"[{i+1}]"
        count    = collect_counts[cls]

        # row background when active
        if is_active:
            blink = int(now*6) % 2 == 0
            bg    = REC_CLR if blink else (80, 50, 180)
            cv2.rectangle(frame, (px+4, ry-2), (px+p_w-4, ry+26), bg, -1)

        cv2.putText(frame, key_str,
                    (px+10, ry+16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42,
                    WHITE if is_active else GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, cls.upper(),
                    (px+38, ry+16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42,
                    WHITE if is_active else GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, str(count),
                    (px+p_w-50, ry+16),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42,
                    ACCENT if is_active else GRAY, 1, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera"); exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time         = time.time()
fps               = 0.0
frame_idx         = 0
held_key          = None   # 1-indexed class currently being held

print("Running.")
print("Hold 1 = collect CAT | Hold 2 = collect PERSON | Hold 3 = collect HAND")
print("G = glow | D = MP labels | Q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame"); break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]
    now   = time.time()

    draw_scan_line(frame, now)

    # ── RGB copy for MediaPipe ───────────────
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ════════════════════════════════════════
    #  [1] OBJECT DETECTION
    # ════════════════════════════════════════
    mp_img     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    det_result = mp_detector.detect(mp_img)
    obj_count  = 0

    for det in det_result.detections:
        bb   = det.bounding_box
        x1,y1 = int(bb.origin_x), int(bb.origin_y)
        x2,y2 = int(bb.origin_x+bb.width), int(bb.origin_y+bb.height)

        if x2-x1 < 10 or y2-y1 < 10:
            continue

        mp_cat   = det.categories[0] if det.categories else None
        mp_label = mp_cat.category_name if mp_cat else "?"
        mp_conf  = float(mp_cat.score)  if mp_cat else 0.0

        crop = frame[max(0,y1):min(H,y2), max(0,x1):min(W,x2)]
        if crop.size == 0:
            continue

        q = obj_queues.setdefault(obj_count, deque(maxlen=SMOOTHING_QUEUE))
        tm_label, tm_conf = classify_crop(crop, obj_infer, obj_labels, q)

        # ── collect cat (key 1) or person (key 2) ──
        is_collecting = False
        if held_key in (1, 2):
            target_cls = COLLECT_CLASSES[held_key-1]   # "cat" or "person"
            if now - last_capture_time >= CAPTURE_INTERVAL:
                collect_counts[target_cls] = save_crop(crop, target_cls)
                last_capture_time = now
                is_collecting = True

        frame = draw_object_box(frame, x1,y1,x2,y2,
                                tm_label, tm_conf,
                                mp_label, mp_conf,
                                obj_count, is_collecting)
        obj_count += 1

    # clean old queues
    for k in list(obj_queues.keys()):
        if k >= obj_count:
            del obj_queues[k]

    # ════════════════════════════════════════
    #  [2] HAND TRACKING + GESTURE
    # ════════════════════════════════════════
    hand_results = hands_det.process(rgb)
    hand_count   = 0

    if hand_results.multi_hand_landmarks:
        for hi, hand_lm in enumerate(hand_results.multi_hand_landmarks):
            lm  = hand_lm.landmark
            pts = [(int(lm[i].x*W), int(lm[i].y*H)) for i in range(21)]
            xs,ys = [p[0] for p in pts],[p[1] for p in pts]
            pad   = 20
            hx1   = max(0, min(xs)-pad)
            hy1   = max(0, min(ys)-pad)
            hx2   = min(W, max(xs)+pad)
            hy2   = min(H, max(ys)+pad)

            hand_crop = frame[hy1:hy2, hx1:hx2]
            if hand_crop.size == 0:
                continue

            q = hand_queues.setdefault(hi, deque(maxlen=SMOOTHING_QUEUE))
            gesture_label, gesture_conf = classify_crop(
                hand_crop, hand_infer, hand_labels, q)

            # ── collect hand (key 3) ──
            is_collecting = False
            if held_key == 3:
                if now - last_capture_time >= CAPTURE_INTERVAL:
                    collect_counts["hand"] = save_crop(hand_crop, "hand")
                    last_capture_time = now
                    is_collecting = True

            frame, _ = draw_hand(frame, hand_lm, W, H, hi,
                                  gesture_label, gesture_conf,
                                  is_collecting)
            hand_count += 1

    for k in list(hand_queues.keys()):
        if k >= hand_count:
            del hand_queues[k]

    # ════════════════════════════════════════
    #  TOP STATUS PANEL
    # ════════════════════════════════════════
    frame = draw_panel(frame, 16, 16, 320, 52)
    cv2.line(frame, (26,16), (326,16), ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, "HUD VISION SUITE",
                (28, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.40, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"OBJ: {obj_count}   HANDS: {hand_count}",
                (28, 54),
                cv2.FONT_HERSHEY_DUPLEX, 0.50, WHITE, 1, cv2.LINE_AA)

    # ════════════════════════════════════════
    #  DATA COLLECTOR PANEL (bottom-right)
    # ════════════════════════════════════════
    frame = draw_collector_panel(frame, held_key, now)

    # ── Bottom hint bar ──────────────────────
    hints = "Hold 1:cat  2:person  3:hand  |  G:glow  D:labels  Q:quit"
    cv2.putText(frame, hints, (16, H-14),
                cv2.FONT_HERSHEY_DUPLEX, 0.35, GRAY, 1, cv2.LINE_AA)
    draw_fps(frame, fps)

    # Live blink dot
    dot_r = 5 if (frame_idx//15)%2==0 else 3
    cv2.circle(frame, (W-90, H-18), dot_r, ACCENT, -1, cv2.LINE_AA)

    # ── FPS ──────────────────────────────────
    fps       = 0.9*fps + 0.1*(1.0/max(now-prev_time, 1e-5))
    prev_time = now

    cv2.imshow("HUD Vision Suite", frame)
    frame_idx += 1

    # ── Key handling ─────────────────────────
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('g'):
        GLOW_ENABLED = not GLOW_ENABLED
    elif key == ord('d'):
        SHOW_MP_LABEL = not SHOW_MP_LABEL
    elif key in (ord('1'), ord('2'), ord('3')):
        held_key = int(chr(key))
    else:
        # Any other key (or no key pressed = 255) releases hold
        if key != 255:
            held_key = None

    # Release held key when nothing is pressed
    # (waitKey returns 255 when no key is held)
    if key == 255:
        held_key = None

cap.release()
hands_det.close()
mp_detector.close()
cv2.destroyAllWindows()

print("\n── Collection summary ──")
for cls in COLLECT_CLASSES:
    print(f"  {cls}: {collect_counts[cls]} images  →  "
          f"{os.path.join(COLLECT_ROOT, cls)}")