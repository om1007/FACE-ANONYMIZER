import cv2
from cvzone.FaceDetectionModule import FaceDetector

# Create a VideoCapture object to access the camera
cap = cv2.VideoCapture(0)  # 0 usually accesses the default camera

detector = FaceDetector(minDetectionCon=0.75)

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    img,bboxs = detector.findFaces(img,draw=False)

    if bboxs:
        for i,bbox in enumerate(bboxs):
            x,y,w,h = bbox['bbox']

            if x < 0: x = 0
            if y < 0: x = 0

            imgCrop = img[y:y + h, x:x + w]
            imgBlur = cv2.blur(imgCrop, (63,63))
            img[y:y + h, x:x + w] = imgBlur
            # cv2.imshow(f'Image Cropped {i}', imgCrop)

    
    # Display the resulting frame
    cv2.imshow('Camera Preview', img)

    # Exit when the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
