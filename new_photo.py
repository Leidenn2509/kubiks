import cv2
from RubiksCube import RubiksCube

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')
saves = []
rb = RubiksCube()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = rb.analyze(frame)
    cv2.imshow('frame', frame)
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        saves.append(frame)

i = 0
for frame in saves:
    cv2.imwrite(str(i) + ".png", frame)
    i += 1
cap.release()
cv2.destroyAllWindows()
