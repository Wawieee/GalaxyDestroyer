"""
Hand Gesture Recognition  ·  HUD Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Dependencies:
    pip install opencv-python mediapipe tensorflow numpy

Controls:  Q  →  quit
"""

import cv2
import numpy as np
import tf_keras as keras
from collections import deque
import time
import mediapipe as mp
import tensorflow as tf

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH   = r"C:\Users\palaw\Documents\Projects\handsign\model\model.savedmodel"
LABELS_PATH  = r"C:\Users\palaw\Documents\Projects\handsign\model\labels.txt"

IMG_SIZE        = 224
SMOOTHING_QUEUE = 8

# Palette (BGR)
ACCENT      = (0, 230, 180)      # teal-mint
ACCENT_DIM  = (0, 130, 100)
FINGER_CLR  = (0, 210, 255)      # cyan for bones
JOINT_CLR   = (255, 255, 255)    # white knuckle dots
TIP_CLR     = (60, 200, 255)     # fingertip dots
TRAIL_CLR   = (0, 180, 120)      # wrist motion trail
BG_DARK     = (18, 18, 24)
WHITE       = (240, 240, 240)
GRAY        = (100, 105, 115)

GLOW_PASSES = 3      # blur passes for glow effect
GLOW_ENABLED = False
# ─────────────────────────────────────────────
#  MediaPipe Hands
# ─────────────────────────────────────────────
mp_hands  = mp.solutions.hands
hands_det = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)
FINGERTIPS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips

# ─────────────────────────────────────────────
#  LOAD MODEL & LABELS
# ─────────────────────────────────────────────
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]

with open(LABELS_PATH, "r") as f:
    labels = [" ".join(line.strip().split()[1:]) for line in f.readlines()]

prediction_queue = deque(maxlen=SMOOTHING_QUEUE)


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────

def blend(frame, overlay, alpha=0.55):
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_rounded_rect(img, x1, y1, x2, y2, r, color, thickness=-1):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, thickness)


def draw_corner_brackets(img, x1, y1, x2, y2, color, length=28, thickness=2):
    for tip, h, v in [
        ((x1, y1), (x1+length, y1), (x1, y1+length)),
        ((x2, y1), (x2-length, y1), (x2, y1+length)),
        ((x1, y2), (x1+length, y2), (x1, y2-length)),
        ((x2, y2), (x2-length, y2), (x2, y2-length)),
    ]:
        cv2.line(img, tip, h, color, thickness, cv2.LINE_AA)
        cv2.line(img, tip, v, color, thickness, cv2.LINE_AA)


def draw_panel(frame, x, y, w, h):
    overlay = frame.copy()
    draw_rounded_rect(overlay, x, y, x+w, y+h, 10, BG_DARK, -1)
    return blend(frame, overlay, alpha=0.72)


def draw_confidence_bars(frame, avg_preds, labels, top_x, top_y, bar_w, row_h=28):
    top_n  = min(3, len(labels))
    ranked = np.argsort(avg_preds)[::-1][:top_n]
    for i, idx in enumerate(ranked):
        conf  = float(avg_preds[idx])
        label = labels[idx]
        y     = top_y + i * row_h
        cv2.putText(frame, label, (top_x, y+14),
                    cv2.FONT_HERSHEY_DUPLEX, 0.42, GRAY, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (top_x, y+18), (top_x+bar_w, y+24), (40,42,48), -1)
        fill = int(conf * bar_w)
        if fill > 0:
            cv2.rectangle(frame, (top_x, y+18), (top_x+fill, y+24),
                          ACCENT if i == 0 else ACCENT_DIM, -1)
        cv2.putText(frame, f"{conf*100:.1f}%", (top_x+bar_w+6, y+24),
                    cv2.FONT_HERSHEY_DUPLEX, 0.38,
                    WHITE if i == 0 else GRAY, 1, cv2.LINE_AA)


def draw_fps(frame, fps):
    cv2.putText(frame, f"FPS {fps:.0f}",
                (frame.shape[1]-80, frame.shape[0]-14),
                cv2.FONT_HERSHEY_DUPLEX, 0.42, GRAY, 1, cv2.LINE_AA)


def draw_scan_line(frame, t):
    y  = int((t * 80) % frame.shape[0])
    ov = frame.copy()
    cv2.line(ov, (0, y), (frame.shape[1], y), ACCENT, 1)
    cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)


# ─────────────────────────────────────────────
#  GLOW PRIMITIVES
# ─────────────────────────────────────────────

def glow_line(frame, pt1, pt2, color, thickness=2):
    if GLOW_ENABLED:
        glow = np.zeros_like(frame)
        cv2.line(glow, pt1, pt2, color, thickness + GLOW_PASSES*2, cv2.LINE_AA)
        glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=3)
        cv2.addWeighted(glow, 0.55, frame, 1.0, 0, frame)
    cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)


def glow_circle(frame, center, radius, color):
    if GLOW_ENABLED:
        glow = np.zeros_like(frame)
        cv2.circle(glow, center, radius + GLOW_PASSES*2, color, -1)
        glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=4)
        cv2.addWeighted(glow, 0.5, frame, 1.0, 0, frame)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)


# ─────────────────────────────────────────────
#  HAND SKELETON
# ─────────────────────────────────────────────

def draw_hand_skeleton(frame, landmarks, W, H, hand_idx):
    lm  = landmarks.landmark
    pts = [(int(lm[i].x * W), int(lm[i].y * H)) for i in range(21)]

    # ── Bones ────────────────────────────────────
    for a, b in CONNECTIONS:
        glow_line(frame, pts[a], pts[b], FINGER_CLR, thickness=2)

    # ── Joints & fingertips ───────────────────────
    FINGER_NAMES = {4:"T", 8:"I", 12:"M", 16:"R", 20:"P"}
    for i, pt in enumerate(pts):
        if i in FINGERTIPS:
            glow_circle(frame, pt, 7, TIP_CLR)
            cv2.circle(frame, pt, 3, WHITE, -1, cv2.LINE_AA)
            # Finger letter label
            name = FINGER_NAMES[i]
            cv2.putText(frame, name, (pt[0]+8, pt[1]-8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.32, TIP_CLR, 1, cv2.LINE_AA)
        elif i == 0:
            glow_circle(frame, pt, 6, ACCENT)
        else:
            cv2.circle(frame, pt, 3, JOINT_CLR, -1, cv2.LINE_AA)

    # ── Hand bounding box brackets ────────────────
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad    = 20
    bx1    = max(0, min(xs) - pad)
    by1    = max(0, min(ys) - pad)
    bx2    = min(W, max(xs) + pad)
    by2    = min(H, max(ys) + pad)
    draw_corner_brackets(frame, bx1, by1, bx2, by2, ACCENT, length=18, thickness=1)


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera"); exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = time.time()
fps       = 0.0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame"); break

    frame = cv2.flip(frame, 1)

    H, W = frame.shape[:2]
    now  = time.time()

    # ── Scan-line ────────────────────────────────
    draw_scan_line(frame, now)

    # ── MediaPipe detection ──────────────────────
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results    = hands_det.process(rgb)
    hands_found = 0

    if results.multi_hand_landmarks:
        hands_found = len(results.multi_hand_landmarks)
        for hi, hand_lm in enumerate(results.multi_hand_landmarks):
            draw_hand_skeleton(frame, hand_lm, W, H, hi)

    # ── Gesture model inference ──────────────────
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    preds_dict  = infer(tf.constant(img))
    preds_array = list(preds_dict.values())[0].numpy()
    prediction_queue.append(preds_array[0])
    avg_preds   = np.mean(prediction_queue, axis=0)
    class_index = int(np.argmax(avg_preds))
    class_label = labels[class_index]
    confidence  = float(avg_preds[class_index])

    # ── FPS ──────────────────────────────────────
    fps       = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-5))
    prev_time = now

    # ════════════════════════════════════════════
    #  TOP PANEL — main prediction
    # ════════════════════════════════════════════
    p_h = 78
    p_w = min(520, W - 20)
    p_x, p_y = 16, 16
    frame = draw_panel(frame, p_x, p_y, p_w, p_h)

    cv2.line(frame, (p_x+10, p_y), (p_x+p_w-10, p_y), ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, "GESTURE DETECTED",
                (p_x+14, p_y+18),
                cv2.FONT_HERSHEY_DUPLEX, 0.38, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(frame, class_label.upper(),
                (p_x+14, p_y+52),
                cv2.FONT_HERSHEY_DUPLEX, 1.05, WHITE, 2, cv2.LINE_AA)

    conf_str = f"{confidence*100:.1f}%"
    (tw, _), _ = cv2.getTextSize(conf_str, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
    cv2.putText(frame, conf_str,
                (p_x+p_w-tw-14, p_y+52),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, ACCENT, 1, cv2.LINE_AA)

    bx1, by1 = p_x+14, p_y+62
    bx2, by2 = p_x+p_w-14, p_y+68
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (40,42,48), -1)
    fw = int(confidence * (bx2 - bx1))
    if fw > 0:
        cv2.rectangle(frame, (bx1, by1), (bx1+fw, by2), ACCENT, -1)

    # ════════════════════════════════════════════
    #  SIDE PANEL — top-3 breakdown
    # ════════════════════════════════════════════
    sp_w = 220
    sp_h = 24 + min(3, len(labels)) * 28 + 8
    sp_x = W - sp_w - 16
    sp_y = 16
    frame = draw_panel(frame, sp_x, sp_y, sp_w, sp_h)
    cv2.line(frame, (sp_x+10, sp_y), (sp_x+sp_w-10, sp_y), ACCENT_DIM, 1, cv2.LINE_AA)
    cv2.putText(frame, "TOP PREDICTIONS",
                (sp_x+10, sp_y+16),
                cv2.FONT_HERSHEY_DUPLEX, 0.36, ACCENT, 1, cv2.LINE_AA)
    draw_confidence_bars(frame, avg_preds, labels,
                         top_x=sp_x+10, top_y=sp_y+22, bar_w=sp_w-70)

    # ════════════════════════════════════════════
    #  BOTTOM — hand count + FPS + live dot
    # ════════════════════════════════════════════
    cv2.putText(frame, f"HANDS  {hands_found}", (16, H-14),
                cv2.FONT_HERSHEY_DUPLEX, 0.42,
                ACCENT if hands_found > 0 else GRAY, 1, cv2.LINE_AA)

    draw_fps(frame, fps)

    dot_r = 5 if (frame_idx // 15) % 2 == 0 else 4
    cv2.circle(frame, (W-90, H-18), dot_r, ACCENT, -1, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)
    frame_idx += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('g'):
        GLOW_ENABLED = not GLOW_ENABLED

cap.release()
hands_det.close()
cv2.destroyAllWindows()