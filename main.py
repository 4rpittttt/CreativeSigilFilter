import cv2
import numpy as np
import mediapipe as mp
import math
import time
import sys

# =====================================================
# 1. WEBCAM — SIMPLE, VERIFIED, STABLE
# =====================================================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
time.sleep(0.5)

if not cap.isOpened():
    print("❌ Webcam not accessible")
    sys.exit(1)

ret, _ = cap.read()
if not ret:
    print("❌ Webcam opened but no frames")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# =====================================================
# 2. MEDIAPIPE HANDS
# =====================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================================================
# 3. COLORS
# =====================================================
CYAN   = (255, 200, 50)
GOLD   = (0, 220, 255)
ORANGE = (0, 140, 255)
WHITE  = (255, 255, 255)

start_time = time.time()

# =====================================================
# 4. HAND UTILITIES
# =====================================================
def palm_center(hand, w, h):
    ids = [0, 5, 9, 13, 17]
    cx = sum(int(hand.landmark[i].x * w) for i in ids) // len(ids)
    cy = sum(int(hand.landmark[i].y * h) for i in ids) // len(ids)
    return cx, cy

def finger_open_ratio(hand, w, h):
    palm = hand.landmark[0]
    tips = [8, 12, 16, 20]
    d = 0
    for t in tips:
        lm = hand.landmark[t]
        d += math.hypot((lm.x - palm.x) * w, (lm.y - palm.y) * h)
    return d / len(tips)

def palm_direction(hand):
    wrist = hand.landmark[0]
    mid = hand.landmark[9]
    return math.atan2(mid.y - wrist.y, mid.x - wrist.x)

# =====================================================
# 5. GEOMETRY SYSTEMS (VERTEX-DRIVEN)
# =====================================================
def sacred_lattice(img, cx, cy, t, scale):
    pts = []
    for i in range(24):
        th = i * math.pi / 12 + t * 0.5
        r = scale * (0.6 + 0.4 * math.sin(t + i))
        pts.append((cx + r * math.cos(th), cy + r * math.sin(th)))

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(i - j) <= 3:
                cv2.line(
                    img,
                    (int(pts[i][0]), int(pts[i][1])),
                    (int(pts[j][0]), int(pts[j][1])),
                    GOLD, 1
                )

def chaotic_core(img, cx, cy, t, scale):
    prev = None
    for i in range(700):
        a = i * 0.045
        r = scale * math.sin(3 * a + t) * math.cos(2 * a)
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        if prev:
            cv2.line(img, prev, (int(x), int(y)), ORANGE, 1)
        prev = (int(x), int(y))

def merged_field(img, cx, cy, t, scale):
    pts = []
    for i in range(64):
        a = i * math.pi / 32
        r = scale * (0.7 + 0.3 * math.sin(t * 1.2 + i))
        pts.append((cx + r * math.cos(a + t),
                    cy + r * math.sin(a - t)))

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if (i + j) % 6 == 0:
                cv2.line(
                    img,
                    (int(pts[i][0]), int(pts[i][1])),
                    (int(pts[j][0]), int(pts[j][1])),
                    CYAN, 1
                )

def energy_ball(img, cx, cy, t, scale):
    # surface particles
    for i in range(220):
        a = i * math.pi / 110
        r = scale * (0.8 + 0.2 * math.sin(t * 3 + i))
        x = cx + r * math.cos(a)
        y = cy + r * math.sin(a)
        cv2.circle(img, (int(x), int(y)), 1, WHITE, -1)

    # internal field lines
    for i in range(48):
        th = i * math.pi / 24 + t
        x = cx + scale * 0.5 * math.cos(th)
        y = cy + scale * 0.5 * math.sin(th)
        cv2.line(img, (cx, cy), (int(x), int(y)), CYAN, 1)

# =====================================================
# 6. MAIN LOOP
# =====================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    layer = np.zeros_like(frame)
    t = time.time() - start_time

    hands_data = []

    if res.multi_hand_landmarks:
        for hand in res.multi_hand_landmarks:
            cx, cy = palm_center(hand, w, h)
            openness = finger_open_ratio(hand, w, h)
            angle = palm_direction(hand)
            hands_data.append((cx, cy, openness, angle))

    if len(hands_data) == 1:
        cx, cy, open_ratio, _ = hands_data[0]
        if open_ratio > 160:
            sacred_lattice(layer, cx, cy, t, 130)
        else:
            chaotic_core(layer, cx, cy, t, 130)

    elif len(hands_data) == 2:
        (x1, y1, o1, a1), (x2, y2, o2, a2) = hands_data
        dist = math.hypot(x1 - x2, y1 - y2)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2

        # Kamehameha pose
        if 120 < dist < 280 and abs(a1 - a2) > 2.2:
            energy_ball(layer, mx, my, t, dist * 0.5)
        elif dist < 150:
            merged_field(layer, mx, my, t, dist)
        else:
            sacred_lattice(layer, x1, y1, t, 110)
            sacred_lattice(layer, x2, y2, t, 110)

    # =================================================
    # 7. GLOW
    # =================================================
    glow = cv2.GaussianBlur(layer, (81, 81), 0)
    output = cv2.addWeighted(frame, 0.85, glow, 0.7, 0)
    output = cv2.addWeighted(output, 1.0, layer, 1.0, 0)

    cv2.imshow("Procedural Energy Geometry", output)

    if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break

# =====================================================
# 8. CLEAN EXIT
# =====================================================
cap.release()
cv2.destroyAllWindows()

