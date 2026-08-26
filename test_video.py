import cv2

cap = cv2.VideoCapture("Demo1.mp4")

count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    count += 1

    if count % 500 == 0:
        print("Read:", count)

cap.release()

print("TOTAL FRAMES READ:", count)