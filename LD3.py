import cv2
import numpy as np

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def darken(img):
    return cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, 0)

def isolate_yellow_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower = np.array([20, 100, 100])
    upper = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hls, lower, upper)
    return yellow_mask

def isolate_white_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    lower = np.array([0, 200, 0])
    upper = np.array([255, 255, 255])
    white_mask = cv2.inRange(hls, lower, upper)
    return white_mask

def combine_masks(mask1, mask2):
    return cv2.bitwise_or(mask1, mask2)

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny_edge(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

def consolidate_lines(img, lines):
    left_lines = []
    right_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
    return left_lines, right_lines

def extrapolate_lines(img, lines):
    ys = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            ys.extend([y1, y2])
    min_y = min(ys)
    max_y = img.shape[0]
    new_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 == x1:
                continue  # Skip vertical lines
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 1e-6:  # Avoid division by zero or infinite slope
                continue
            intercept = y1 - slope * x1
            x1_new = int((min_y - intercept) / slope)
            x2_new = int((max_y - intercept) / slope)
            new_lines.append([[x1_new, min_y, x2_new, max_y]])
    return new_lines

def draw_lane_lines(img, lines):
    line_img = np.zeros_like(img)
    draw_lines(line_img, lines)
    return cv2.addWeighted(img, 0.8, line_img, 1, 0)

# Process video
def process_video(input_file, output_file):
    cap = cv2.VideoCapture(input_file)
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'XVID'), 25, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess the frame
        gray = grayscale(frame)
        darkened_gray = darken(gray)
        yellow_mask = isolate_yellow_hls(frame)
        white_mask = isolate_white_hls(frame)
        combined_mask = combine_masks(yellow_mask, white_mask)
        masked_img = apply_mask(darkened_gray, combined_mask)
        blurred_img = gaussian_blur(masked_img, 5)
        edges = canny_edge(blurred_img, 50, 150)
        
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

        out.write(lane_image)

        cv2.imshow('Lane Detection', lane_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Process the video file
process_video("D:/thesis/example1.mp4", "D:/thesis/outexp.mp4")
