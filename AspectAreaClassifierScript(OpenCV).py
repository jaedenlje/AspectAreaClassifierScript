import cv2
import numpy as np
import os

# Load the user-defined circular reference image
ref_image = cv2.imread('/path/to/your/shape/specimen/image', 0)

# Define the contour of the reference image
ref_contour = cv2.findContours(ref_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

# Load the test image folder
test_folder = "/path/to/your/test/folder"

# Loop through all images in the test folder
for filename in os.listdir(test_folder):
    # Load the image
    image_path = os.path.join(test_folder, filename)
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment out the shapes
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # Classify arrow vs round shape using circular line computer vision detection
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        print(f'Aspect Ratio: {aspect_ratio}, Area: {area}')

        # Check if the contour is circular
        if aspect_ratio > 0.5:
            shape = 'Circular'
            print('Circular shape detected')
        else:
            shape = 'Arrow'
            print('Arrow shape detected')

        # Crop the shape
        x, y, w, h = cv2.boundingRect(contour)
        crop = image[y:y+h, x:x+w]

        # Classify the color (for round traffic light)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Define the color ranges for green, yellow, and red
        green_range = (70, 90, 100), (100, 120, 140)
        yellow_range = (20, 100, 100), (30, 120, 120)
        red_range = (0, 100, 100), (10, 120, 120)

        # Create masks for each color
        green_mask = cv2.inRange(hsv, *green_range)
        yellow_mask = cv2.inRange(hsv, *yellow_range)
        red_mask = cv2.inRange(hsv, *red_range)

        # Count the pixels for each color
        green_pixels = cv2.countNonZero(green_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        red_pixels = cv2.countNonZero(red_mask)

        # Check if the cropped shape is green
        if green_pixels > yellow_pixels and green_pixels > red_pixels:
            color = 'Green'
            print(f"Green {shape} detected in {filename}")
        elif yellow_pixels > green_pixels and yellow_pixels > red_pixels:
            color = 'Yellow'
            print(f"Yellow {shape} detected in {filename}")
        else:
            color = 'Red'
            print(f"Red {shape} detected in {filename}")

    # Resize the green mask to match the image dimensions
    green_mask_resized = cv2.resize(green_mask, (image.shape[1], image.shape[0]))

    # Display the original image and the masked image side by side
    combined_image = np.hstack([image, cv2.cvtColor(green_mask_resized, cv2.COLOR_GRAY2BGR)])
    cv2.imshow('Image and Mask', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
