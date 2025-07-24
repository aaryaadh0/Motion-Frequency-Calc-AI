import cv2

# Default webcam open garne (0 means built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Frame window ma dekhaune
    cv2.imshow('Webcam Feed', frame)

    # 'q' key thichda loop bata niskine
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release garne resources
cap.release()
cv2.destroyAllWindows()