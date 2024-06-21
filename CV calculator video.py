import cv2 as cv
import numpy as np

# Capture video from the camera (0 corresponds to the default camera)
cap = cv.VideoCapture(1)

# List to store droplet sizes
droplet_sizes = []

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

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

            # Append the radius of the detected droplet to the list
            droplet_sizes.append(i[2])

    # Display the live video with detected circles
    cv.imshow("Circles Detected", frame)

    # Break the loop when 'q' key is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate the coefficient of variation if droplets were detected
if droplet_sizes:
    cv_value = np.std(droplet_sizes) / np.mean(droplet_sizes) * 100
    print('Coefficient of Variation of droplets:', cv_value)
else:
    print("No droplets detected")

# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()


# Release the video capture object and close all windows
cap.release()
cv.destroyAllWindows()
