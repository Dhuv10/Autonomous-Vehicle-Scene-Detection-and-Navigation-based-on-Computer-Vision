import numpy as np 
import cv2 
from keras.models import load_model
from ultralytics import YOLO  
from ultralytics.utils.plotting import Annotator, colors  
from collections import defaultdict  

# Load YOLOv5 model
model = YOLO("yolov8s.pt")  # Initialize YOLO object detector with the YOLOv5 model
names = model.model.names  # Get the class names of the YOLO model

# Load Traffic Sign Detection Model
model_traffic_sign = load_model("C:/Users/Rahul/TrafficSignDetectormodel.h5")  # Load the pre-trained traffic sign detection model

# Initialize a dictionary to store the object tracking history
track_history = defaultdict(lambda: [])  # Create a dictionary to store tracking history for each object

# Function for preprocessing traffic sign detection
def preprocess_traffic_sign(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    img = cv2.equalizeHist(img)  # Apply histogram equalization for contrast enhancement
    img = img / 255  # Normalize pixel values to the range [0, 1]
    img = cv2.resize(img, (32, 32))  # Resize image to the required input size of the model
    img = np.expand_dims(img, axis=-1)  # Add an extra dimension to represent batch size
    return img

# Lane detection functions

# Function to convert an image to grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Function to darken the image
def darken(img):
    return cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, 0)

# Function to isolate yellow regions in the image using HLS color space
def isolate_yellow_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # Convert image to HLS color space
    lower = np.array([20, 100, 100])  # Define lower threshold for yellow color
    upper = np.array([30, 255, 255])  # Define upper threshold for yellow color
    yellow_mask = cv2.inRange(hls, lower, upper)  # Create mask to isolate yellow regions
    return yellow_mask

# Function to isolate white regions in the image using HLS color space
def isolate_white_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # Convert image to HLS color space
    lower = np.array([0, 200, 0])  # Define lower threshold for white color
    upper = np.array([255, 255, 255])  # Define upper threshold for white color
    white_mask = cv2.inRange(hls, lower, upper)  # Create mask to isolate white regions
    return white_mask

# Function to combine two binary masks using bitwise OR operation
def combine_masks(mask1, mask2):
    return cv2.bitwise_or(mask1, mask2)

# Function to apply a mask to the image
def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

# Function to apply Gaussian blur to the image
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Function to perform Canny edge detection on the image
def canny_edge(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Function to define region of interest (ROI) in the image
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)  # Create a mask of zeros with the same shape as the input image
    cv2.fillPoly(mask, vertices, 255)  # Fill the polygon defined by the vertices with white color
    masked_img = cv2.bitwise_and(img, mask)  # Apply the mask to the input image
    return masked_img

# Function to draw lines on the image
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)  # Draw a line on the image

# Function to detect lines using Hough transform
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Function to consolidate lines into left and right lane lines
def consolidate_lines(img, lines):
    left_lines = []
    right_lines = []
    if lines is not None:  # Check if lines are detected
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)  # Calculate slope of the line
                if slope < 0:  # If slope is negative, it belongs to the left lane
                    left_lines.append(line)
                else:  # If slope is positive, it belongs to the right lane
                    right_lines.append(line)
    return left_lines, right_lines

# Function to extrapolate lines to cover the full lane
def extrapolate_lines(img, lines):
    ys = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            ys.extend([y1, y2])  # Collect y-coordinates of line endpoints
    if not ys:  # Check if ys is empty
        return []  # Return an empty list if no lines are detected
    min_y = min(ys)  # Find the minimum y-coordinate
    max_y = img.shape[0]  # Get the height of the image
    new_lines = []  # Initialize list to store extrapolated lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 == x1:
                continue  # Skip vertical lines
            slope = (y2 - y1) / (x2 - x1)  # Calculate slope of the line
            if abs(slope) < 1e-6:  # Avoid division by zero or infinite slope
                continue
            intercept = y1 - slope * x1  # Calculate y-intercept of the line
            x1_new = int((min_y - intercept) / slope)  # Calculate new x-coordinate for bottom of the image
            x2_new = int            ((max_y - intercept) / slope)  # Calculate new x-coordinate for top of the image
            new_lines.append([[x1_new, min_y, x2_new, max_y]])  # Append extrapolated line coordinates to the list
    return new_lines  # Return the list of extrapolated lines

# Function to draw lane lines on the image
def draw_lane_lines(img, lines):
    line_img = np.zeros_like(img)  # Create a blank image with the same size as the input image
    draw_lines(line_img, lines, color=[255, 0, 0], thickness=10)  # Draw thick blue lines representing lane boundaries
    return cv2.addWeighted(img, 1, line_img, 0.5, 0)  # Overlay the lane lines on the original image with reduced opacity

# Function to darken the frame
def darken_frame(frame, factor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert BGR image to HSV color space
    h, s, v = cv2.split(hsv)  # Split the HSV image into individual channels
    v = np.clip(v * factor, 0, 255).astype(np.uint8)  # Adjust brightness by scaling the V channel
    darkened_hsv = cv2.merge([h, s, v])  # Merge the adjusted channels back into an HSV image
    darkened_frame = cv2.cvtColor(darkened_hsv, cv2.COLOR_HSV2BGR)  # Convert the darkened HSV image back to BGR color space
    return darkened_frame  # Return the darkened frame

# Path to the input video file
video_path = "D:/thesis/example2.mp4"
cap = cv2.VideoCapture(video_path)  # Open the video file for reading
assert cap.isOpened(), "Error reading video file"  # Check if the video file is opened successfully

# Get video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize VideoWriter for output
result = cv2.VideoWriter("object_detection_tracking_lane_detection_and_traffic_sign_detection.avi",
                         cv2.VideoWriter_fourcc(*'XVID'),
                         fps,
                         (w, h))

while cap.isOpened():
    success, frame = cap.read()  # Read a frame from the video
    if success:
        # Perform object detection and tracking
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if boxes.numel() > 0:  # Check if any bounding boxes are detected
            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Initialize Annotator for drawing on the frame
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id, conf in zip(boxes, clss, track_ids, confs):
                # Draw bounding box around detected traffic sign
                label_text = f"{names[int(cls)]} {conf:.2f} {track_id}"
                annotator.box_label(box, color=colors(int(cls), True), label=label_text)
                if names[int(cls)] == "traffic sign":  # Check if the detected object is a traffic sign
                    # Extract region for traffic sign detection
                    x1, y1, x2, y2 = [int(i) for i in box]
                    roi_traffic_sign = frame[y1:y2, x1:x2]
                    # Preprocess frame for traffic sign detection
                    preprocessed_traffic_sign = preprocess_traffic_sign(roi_traffic_sign)
                    # Predict using traffic sign detection model
                    prediction = model_traffic_sign.predict(np.array([preprocessed_traffic_sign]))
                    predicted_class = np.argmax(prediction)
                    # Draw traffic sign label
                    cv2.putText(frame, f"Traffic Sign: {predicted_class}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Perform lane detection
        gray = grayscale(frame)  # Convert frame to grayscale
        darkened_gray = darken(gray)  # Darken the grayscale frame
        yellow_mask = isolate_yellow_hls(frame)  # Isolate yellow regions in the frame
        white_mask = isolate_white_hls(frame)  # Isolate white regions in the frame
        combined_mask = combine_masks(yellow_mask, white_mask)  # Combine yellow and white masks
        masked_img = apply_mask(darkened_gray, combined_mask)  # Apply the combined mask to the darkened grayscale frame
        blurred_img = gaussian_blur(masked_img, 5)  # Apply Gaussian blur to the masked image
        edges = canny_edge(blurred_img, 50, 150)  # Detect edges using Canny edge detection
        
        # Define region of interest
        imshape = frame.shape
        vertices = np.array([[(0, imshape[0]), (650, 490), (650, 490), (imshape[1], imshape[0])]], dtype=np.int32)
        roi_edges = region_of_interest(edges, vertices)

        # Detect lines using Hough transform
        lines = hough_lines(roi_edges, 1, np.pi/180, 50, 100, 160)

        # Consolidate and extrapolate lines
        left_lines, right_lines = consolidate_lines(frame, lines)
        extrapolated_left_lines = extrapolate_lines(frame, left_lines)
        extrapolated_right_lines = extrapolate_lines(frame, right_lines)

        # Draw lane lines on original frame
        lane_lines = extrapolated_left_lines + extrapolated_right_lines
        lane_image = draw_lane_lines(frame, lane_lines)
        
        darkened_frame = darken_frame(frame, 0.2)  # Darken the frame for better visualization
        result_frame = cv2.addWeighted(darkened_frame, 1, lane_image, 0.5, 0)  # Overlay the lane lines on the darkened frame

        # Write annotated frame to output video
        result.write(result_frame)

        # Display annotated frame
        cv2.imshow("Frame", result_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release video capture and VideoWriter objects
result.release()
cap.release()
cv2.destroyAllWindows()


