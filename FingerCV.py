import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands()


TIP_IDS = [4, 8, 12, 16, 20]


def count_fingers(hand_landmarks):
    finger_count = 0
    for id in TIP_IDS:

        if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
            finger_count += 1
    return finger_count


# Відкриття веб-камери
cap = cv2.VideoCapture(0)

while True:
    success, image = cap.read()
    if not success:
        continue


    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


    finger_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_count += count_fingers(hand_landmarks)


            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    print("Кількість пальців:", finger_count)


    cv2.imshow('Finger Counting', image)


    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

