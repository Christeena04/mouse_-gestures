import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Screen size
screen_width, screen_height = pyautogui.size()

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

p_time = 0
click_down = False
screenshot_taken = False

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    frame_height, frame_width, _ = img.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Landmark positions
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]
            middle_tip = hand_landmarks.landmark[12]

            ix, iy = int(index_tip.x * frame_width), int(index_tip.y * frame_height)
            tx, ty = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
            mx, my = int(middle_tip.x * frame_width), int(middle_tip.y * frame_height)

            # Move mouse
            screen_x = np.interp(ix, [0, frame_width], [0, screen_width])
            screen_y = np.interp(iy, [0, frame_height], [0, screen_height])
            pyautogui.moveTo(screen_x, screen_y)

            # Draw circle at index finger
            cv2.circle(img, (ix, iy), 10, (0, 255, 255), cv2.FILLED)

            # Click detection: thumb + index
            click_distance = math.hypot(tx - ix, ty - iy)
            if click_distance < 30:
                cv2.circle(img, ((ix + tx) // 2, (iy + ty) // 2), 15, (0, 0, 255), cv2.FILLED)
                if not click_down:
                    pyautogui.click()
                    click_down = True
            else:
                click_down = False

            # Screenshot detection: index + middle
            screenshot_distance = math.hypot(mx - ix, my - iy)
            if screenshot_distance < 30:
                cv2.circle(img, ((ix + mx) // 2, (iy + my) // 2), 15, (255, 0, 255), cv2.FILLED)
                if not screenshot_taken:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    pyautogui.screenshot(f"screenshot_{timestamp}.png")
                    screenshot_taken = True
            else:
                screenshot_taken = False

    # FPS Display
    c_time = time.time()
    fps = 1 / (c_time - p_time + 1e-5)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show Window
    cv2.imshow("Virtual Mouse with Screenshot", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
