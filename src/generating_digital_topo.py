import cv2
import json
import math

"""
This script allows you to generate a digital topo.
You can map all the holds visible in the image by drawing circles around them and annotate the corresponding score.
The digital topo will then be saved as a json-file and jpg.
"""

# Set the file name (modify this variable to change file paths automatically)
file_name = "edited_villars_men_semifinals_n110_plus_14+"

# Automatically generated paths based on the file name
image_path = f'../data/input/frames/{file_name}.jpg'
annotated_image_path = f'../data/input/topos/{file_name}_annotated.jpg'
annotations_json_path = f'../data/input/topos/{file_name}_annotations.json'

# List to hold circle data
annotations = []

# Variables to store the center and radius for the circle
center_point = None
radius = None

# Define the annotation color (black or white)
# Change the value of color to (0, 0, 0) for black or (255, 255, 255) for white
color = (255, 255, 255)

# Function to handle mouse events
def click_event(event, x, y, flags, param):
    global center_point, radius, annotations, image, display_image, resize_factor

    # Convert the coordinates from the resized display back to the original image size
    x_orig = int(x / resize_factor)
    y_orig = int(y / resize_factor)

    # Left mouse button down
    if event == cv2.EVENT_LBUTTONDOWN:
        center_point = (x_orig, y_orig)  # Store the center point in original coordinates
        radius = None  # Reset the radius

    # Mouse move with left button pressed
    elif event == cv2.EVENT_MOUSEMOVE and center_point is not None:
        # Update radius while dragging the mouse
        radius = int(math.sqrt((x_orig - center_point[0])**2 + (y_orig - center_point[1])**2))
        temp_image = image.copy()  # Use the original image to keep the annotations
        cv2.circle(temp_image, center_point, radius, color, 2)
        cv2.circle(temp_image, center_point, 2, color, -1)  # Draw center point
        temp_display_image = cv2.resize(temp_image, (int(input_width * resize_factor), int(input_height * resize_factor)))  # Resize for display
        cv2.imshow('image', temp_display_image)

    # Left mouse button up
    elif event == cv2.EVENT_LBUTTONUP:
        radius = int(math.sqrt((x_orig - center_point[0])**2 + (y_orig - center_point[1])**2))
        hold_number = input(f"Enter hold number for circle with center {center_point} and radius {radius}: ")

        # Save the circle data
        annotations.append({"number": hold_number, "center": center_point, "radius": radius})

        # Draw the final circle on the full-size image
        cv2.circle(image, center_point, radius, color, 2)
        cv2.circle(image, center_point, 2, color, -1)  # Draw center point

        # Calculate text position (outside the circle)
        angle = math.atan2(-20, 20)  # Example angle for offset direction
        line_end_x = int(center_point[0] + radius * math.cos(angle))
        line_end_y = int(center_point[1] + radius * math.sin(angle))
        text_position = (line_end_x + 10, line_end_y - 10)  # Slight offset for text placement

        # Draw line from circle boundary to text
        cv2.line(image, (line_end_x, line_end_y), text_position, color, 2)

        # Add hold number text
        cv2.putText(image, hold_number, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display the resized image during annotation
        display_image = cv2.resize(image, (int(input_width * resize_factor), int(input_height * resize_factor)))  # Resize for display
        cv2.imshow('image', display_image)

        # Reset center point for the next circle
        center_point = None

# Load the image
image = cv2.imread(image_path)

# Confirm the input image dimensions
if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

input_height, input_width = image.shape[:2]
print(f"Input image dimensions: {input_width} x {input_height}")

# Resize factor for display purposes
resize_factor = 0.6  # Adjust this value as needed for display
display_image = cv2.resize(image, (int(input_width * resize_factor), int(input_height * resize_factor)))

# Display the resized image for annotations
cv2.imshow('image', display_image)

# Set mouse callback
cv2.setMouseCallback('image', click_event)

# Wait until 'e' is pressed to save and exit
while True:
    key = cv2.waitKey(1) & 0xFF  # Wait for a key press
    if key == ord('e'):  # If 'e' is pressed, save the annotations and exit
        # Save annotated image in the same resolution as the original
        cv2.imwrite(annotated_image_path, image)
        print(f"Annotated image saved as {annotated_image_path}")

        # Save annotations to a JSON file
        with open(annotations_json_path, 'w') as f:
            json.dump(annotations, f)
        print(f"Annotations saved to {annotations_json_path}")
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
