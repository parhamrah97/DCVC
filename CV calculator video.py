import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Capture video from the camera (0 corresponds to the default camera)
cap = cv.VideoCapture(1)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Display the live video
    cv.imshow("Live Video", frame)

    # Convert the frame to grayscale
    gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    img2 = cv.medianBlur(gray_img, 5)

    # Detect circles using HoughCircles
    circles = cv.HoughCircles(img2, cv.HOUGH_GRADIENT, 1, 200, param1=60, param2=35, minRadius=20, maxRadius=100)

    if circles is not None:
        # Convert the circle parameters to integers
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # Draw the outer circle
            cv.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 6)
            # Draw the center of the circle
            cv.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        # Display the image with detected circles
        cv.imshow("Circles Detected", frame)

        # Calculate the coefficient of variation
        lst = [r for (_, _, r) in circles[0, :]]
        cv_value = np.std(lst) / np.mean(lst) * 100
        print('Coefficient of variation of droplets: ', cv_value)

    # Break the loop when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()
