import cv2
import numpy as np

# Step 1: Load image
img = cv2.imread("input.jpg")   # make sure input.jpg exists
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Step 2: Create window
cv2.namedWindow("Interactive Color Filter")

# Step 3: Create trackbars for HSV ranges
cv2.createTrackbar("H Min", "Interactive Color Filter", 0, 179, lambda x: None)
cv2.createTrackbar("H Max", "Interactive Color Filter", 179, 179, lambda x: None)
cv2.createTrackbar("S Min", "Interactive Color Filter", 0, 255, lambda x: None)
cv2.createTrackbar("S Max", "Interactive Color Filter", 255, 255, lambda x: None)
cv2.createTrackbar("V Min", "Interactive Color Filter", 0, 255, lambda x: None)
cv2.createTrackbar("V Max", "Interactive Color Filter", 255, 255, lambda x: None)

while True:
    # Step 4: Get values from trackbars
    h_min = cv2.getTrackbarPos("H Min", "Interactive Color Filter")
    h_max = cv2.getTrackbarPos("H Max", "Interactive Color Filter")
    s_min = cv2.getTrackbarPos("S Min", "Interactive Color Filter")
    s_max = cv2.getTrackbarPos("S Max", "Interactive Color Filter")
    v_min = cv2.getTrackbarPos("V Min", "Interactive Color Filter")
    v_max = cv2.getTrackbarPos("V Max", "Interactive Color Filter")

    # Step 5: Create mask based on HSV range
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)

    # Step 6: Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=mask)

    # Step 7: Show results
    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Filtered Result", result)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
