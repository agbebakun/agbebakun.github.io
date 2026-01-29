# Classify image
# By Dan Jackson, 2026

from fileinput import filename
import sys
import cv2
import numpy as np
import torch

from classify import load_model, classify, print_scores, most_likely, print_pairwise_scores

LINE_DIAMETER = 16
PADDING = 16
UNIT_SIZE = 256
CLASSIFY_SIZE = 28
IMAGE_FILTER_COLOR = False  # When image is on a background color
CLASSES_FILE = 'classes.json'
CALIBRATION_FILE = 'calibration.json'

CAPTURE_IMAGE_SHARPEN = False
CAPTURE_IMAGE_PERSPECTIVE = True
CAPTURE_IMAGE_INSET = 0.05        # Default: Additional 5% crop inset margin

# Category-Category axis evaluation data
config = {}

# Capture image coordinates, etc.
# To manually find, open capture.png.orig.png with e.g. Pinta.app to determine crop coordinates
CALIBRATION_NTH_SQUARE = 0  # 0 = largest, -1 = smallest
calibration = {
   # "coordinates": [ [782, 469], [1278, 437], [1298, 907], [825, 932] ], # Top-left, Top-right, Bottom-right, Bottom-left
}


# Skeletonize - from: https://opencvpython.blogspot.com/2012/05/skeletonization-using-opencv-python.html
def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    done = False
    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()
        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True
    return skel


def load_from_buffer(file_buffer):
    file_bytes = np.frombuffer(file_buffer, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)  # preserve any alpha (rather than IMREAD_COLOR)
    return image

def load_from_file(filename):
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    return image

# Normalize image for classification
def normalize_image(image, hack_sharpen = False, debugPrefix = None):
    w, h = image.shape[1], image.shape[0]

    # If has alpha, flatten against a white background
    if image.shape[2] == 4:
        background = (255, 255, 255)
        alpha_channel = image[:, :, 3] / 255.0
        for c in range(3):
            image[:, :, c] = (1.0 - alpha_channel) * background[c] + alpha_channel * image[:, :, c]
        image = image[:, :, 0:3]

    # If not already greyscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        if IMAGE_FILTER_COLOR:
            # Use only the blue channel
            image = image[:, :, 2]
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if debugPrefix:
        cv2.imwrite(debugPrefix + ".grey.png", image)

    # Denoise
    image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)

    # Sharpen
    if hack_sharpen:
        kernel = np.array([[0, -1, 0],
                        [-1, 5,-1],
                        [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        image = cv2.filter2D(image, -1, kernel)  # apply repeatedly

    # Blur
    image = cv2.medianBlur(image, 5)  #image = cv2.GaussianBlur(image, (5,5), 0)
 
    # Dynamic binary thresholding
    #_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    if debugPrefix:
        cv2.imwrite(debugPrefix + ".proc.png", image)

    # Invert image
    image = 255 - image
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".inv.png", image)

    # Skeletonize the image
    image = skeletonize(image)
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".skel.png", image)

    # Remove lone pixels
    kernel = np.array([ [-1, -1, -1],
                        [-1,  1, -1],
                        [-1, -1, -1] ], dtype="int")
    single_pixels = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
    single_pixels_inv = cv2.bitwise_not(single_pixels)
    image = cv2.bitwise_and(image, image, mask=single_pixels_inv)

    # Find current bounding box
    min_y = 0
    max_y = h
    min_x = 0
    max_x = w
    ys, xs = np.nonzero(image)
    if len(xs) > 0:
        min_x = np.min(xs)
        max_x = np.max(xs)
    if len(ys) > 0:
        min_y = np.min(ys)
        max_y = np.max(ys)

    # Adjust so that bounding box has padding, is square, and centred
    box_width = max_x - min_x
    box_height = max_y - min_y
    box_size = max(box_width, box_height)
    if box_size < 1:
        box_size = 4
    box_size += (PADDING * 2 + LINE_DIAMETER) * box_size / UNIT_SIZE
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    min_x = int(center_x - box_size / 2)
    max_x = int(center_x + box_size / 2)
    min_y = int(center_y - box_size / 2)
    max_y = int(center_y + box_size / 2)

    # Crop image, with black where out of bounds
    cropped_image = np.zeros((max_y - min_y, max_x - min_x), dtype=np.uint8)
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            if 0 <= y < h and 0 <= x < w:
                cropped_image[y - min_y, x - min_x] = image[y, x]
    image = cropped_image
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".crop.png", image)

    # Dilate the image to the required line thickness
    dilate_radius = LINE_DIAMETER * box_size / UNIT_SIZE / 2
    kernel_size = int(round(dilate_radius * 2)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image = cv2.dilate(image, kernel)
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".dilate.png", image)

    # Resize
    image = cv2.resize(image, (CLASSIFY_SIZE, CLASSIFY_SIZE), interpolation=cv2.INTER_AREA)
    if debugPrefix:
        cv2.imwrite(debugPrefix + ".norm.png", image)

    # Return
    return image


# Crop equally on all sides by a proportion
def crop_margin_proportion(image, margin_proportion):
    h, w = image.shape[0], image.shape[1]
    margin_x = int(w * margin_proportion)
    margin_y = int(h * margin_proportion)
    image = image[margin_y:h - margin_y, margin_x:w - margin_x]
    return image


# Crop with perspective transform
def crop_image_perspective(image, crop_coordinates):
    h, w = image.shape[0], image.shape[1]
    # Determine ordered coordinates (as they may be in any order)
    ordered_crop = [
        min(crop_coordinates, key=lambda p: p[0] + p[1]), # Top-left
        max(crop_coordinates, key=lambda p: p[0] - p[1]), # Top-right
        max(crop_coordinates, key=lambda p: p[0] + p[1]), # Bottom-right
        min(crop_coordinates, key=lambda p: p[0] - p[1]), # Bottom-left
    ]

    # Determine size of the new image
    width_a = np.sqrt(((ordered_crop[2][0] - ordered_crop[3][0]) ** 2) + ((ordered_crop[2][1] - ordered_crop[3][1]) ** 2))
    width_b = np.sqrt(((ordered_crop[1][0] - ordered_crop[0][0]) ** 2) + ((ordered_crop[1][1] - ordered_crop[0][1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((ordered_crop[1][0] - ordered_crop[2][0]) ** 2) + ((ordered_crop[1][1] - ordered_crop[2][1]) ** 2))
    height_b = np.sqrt(((ordered_crop[0][0] - ordered_crop[3][0]) ** 2) + ((ordered_crop[0][1] - ordered_crop[3][1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Destination coordinates
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Perspective transform
    src = np.array(ordered_crop, dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped


# Crop based on the axis-aligned bounding box of the coordinates
def crop_image_axis_aligned(image, crop_coordinates):
    h, w = image.shape[0], image.shape[1]
    # Determine ordered coordinates (as they may be in any order)
    ordered_crop = [
        min(crop_coordinates, key=lambda p: p[0] + p[1]), # Top-left
        max(crop_coordinates, key=lambda p: p[0] - p[1]), # Top-right
        max(crop_coordinates, key=lambda p: p[0] + p[1]), # Bottom-right
        min(crop_coordinates, key=lambda p: p[0] - p[1]), # Bottom-left
    ]

    # Determine axis-aligned bounding box that's strictly within the crop coordinates
    min_x = max(ordered_crop[0][0], ordered_crop[3][0])
    max_x = min(ordered_crop[1][0], ordered_crop[2][0])
    min_y = max(ordered_crop[0][1], ordered_crop[1][1])
    max_y = min(ordered_crop[2][1], ordered_crop[3][1])

    # Constrain to image bounds
    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, w - 1)
    max_y = min(max_y, h - 1)
    image = image[min_y:max_y, min_x:max_x]

    return image


def evaluate(filenames):
    model = load_model()

    hack_sharpen = False

    # Where no filenames were specified, capture using the camera
    capture_filename = "capture.png"
    if not filenames:
        print("CAMERA: Using camera to capture image...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return
        cv2.waitKey(1000)  # Wait for camera adjustments
        ret, image = cap.read()
        if not ret:
            print("ERROR: Could not read frame from camera.")
            return
        
        cv2.imwrite(capture_filename + '.orig.png', image)
        cap.release()

        capture_coordinates = calibration.get('coordinates', None)
        if capture_coordinates:
            if CAPTURE_IMAGE_PERSPECTIVE:
                image = crop_image_perspective(image, capture_coordinates)
            else:
                image = crop_image_axis_aligned(image, capture_coordinates)
        cv2.imwrite(capture_filename + '.cut.png', image)

        capture_inset = calibration.get('inset', CAPTURE_IMAGE_INSET)
        if capture_inset:
            image = crop_margin_proportion(image, capture_inset)
        cv2.imwrite(capture_filename, image)

        filenames = [capture_filename]

    # Evaluate each image from the command line
    for filename in filenames:
        # HACK: Detect captured images only based on filename
        is_capture = (filename == capture_filename)

        image = load_from_file(filename)

        # If image dimensions are CLASSIFY_SIZE x CLASSIFY_SIZE and greyscale, skip normalization
        if image.shape[0] == CLASSIFY_SIZE and image.shape[1] == CLASSIFY_SIZE and image.dtype == np.uint8:
            # if not greyscale
            if len(image.shape) != 2:
                if image.shape[2] == 4:
                    normalized_image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                else:
                    normalized_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                normalized_image = image
            print("NOTE: Skipping normalization step for already-normalized image.")
        else:
            #debugPrefix = None
            debugPrefix = filename

            if is_capture:
                hack_sharpen = CAPTURE_IMAGE_SHARPEN
            else:
                hack_sharpen = False
            
            normalized_image = normalize_image(image, hack_sharpen, debugPrefix)
        
        class_scores = classify(model, normalized_image)
        print()
        print_scores(class_scores)
        detected_class = most_likely(class_scores)

        print_pairwise_scores(class_scores, config.get('pairs', None))

        if is_capture:
            print()
            print(f"CLASSIFICATION: {detected_class}")
            print()
            # Write the result label to a new blank image
            labeled_image = np.ones((100, 1200, 3), dtype=np.uint8) * 255
            cv2.putText(labeled_image, f"Detected: {detected_class}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            cv2.imwrite(capture_filename + ".result.png", labeled_image)

        else:
            print(f"IMAGE: {filename} --> {detected_class}")


def find_squares(image_filename, highlight_nth_square=None):
    # Load image
    image = load_from_file(image_filename)
    orig_image = image.copy()
    h, w = image.shape[0], image.shape[1]

    # Convert to greyscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Blur
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    squares = []
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            squares.append(approx.reshape(4, 2))

    # Sort squares by area (largest first)
    squares.sort(key=lambda sq: cv2.contourArea(sq), reverse=True)

    print(f"SQUARES: Found {len(squares)} squares in image.")

    # Delete squares wholly contained within a larger square
    filtered_squares = []
    for i in range(len(squares)):
        square_i = squares[i]
        area_i = cv2.contourArea(square_i)
        contained = False
        for j in range(len(squares)):
            if i == j:
                continue
            square_j = squares[j]
            area_j = cv2.contourArea(square_j)
            if area_j <= area_i:
                continue
            # Check if all points of square_i are within square_j
            inside_count = 0
            for point in square_i:
                # Construct point
                pt = tuple([int(round(point[0]) ), int(round(point[1]))])
                result = cv2.pointPolygonTest(square_j, pt, False)
                if result >= 0:
                    inside_count += 1
            if inside_count == 4:
                contained = True
                break
        if not contained:
            filtered_squares.append(square_i)

    # Delete squares that have their centres inside a larger square
    final_squares = []
    for i in range(len(filtered_squares)):
        square_i = filtered_squares[i]
        area_i = cv2.contourArea(square_i)
        M = cv2.moments(square_i)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centre_point = (cX, cY)
        contained = False
        for j in range(len(filtered_squares)):
            if i == j:
                continue
            square_j = filtered_squares[j]
            area_j = cv2.contourArea(square_j)
            if area_j <= area_i:
                continue
            result = cv2.pointPolygonTest(square_j, centre_point, False)
            if result >= 0:
                contained = True
                break
        if not contained:
            final_squares.append(square_i)

    print(f"SQUARES: After filtering, {len(final_squares)} squares remain.")

    # Draw detected squares
    for i in range(len(final_squares)):
        square = final_squares[i]
        # Other squares
        color = (0, 255, 0)  # Green
        if i == 0:  # Largest square (nth=0)
            color = (0, 0, 255)  # Red
        elif i == 1: # Second largest square (nth=1)
            color = (255, 0, 0)  # Magenta
        elif i == len(final_squares) - 1:  # Smallest square (nth=-1)
            color = (255, 0, 0)  # Blue

        thickness = 3
        if highlight_nth_square is not None and i == highlight_nth_square:
            color = (0, 255, 255)  # Yellow
            thickness = 5

        cv2.polylines(orig_image, [square], isClosed=True, color=color, thickness=3)
        cv2.putText(orig_image, str(i), tuple(square[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imwrite(image_filename + ".squares.png", orig_image)
    return final_squares


def calibrate(calibration_image=None):
    if calibration_image:
        print(f"CALIBRATION: Using existing image: {calibration_image}")
    else:
        calibration_image = "calibration.png"
        print("CALIBRATION: Capture an image with the camera showing the drawing area.")
        # print("Press Enter to capture...")
        # input()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return
        cv2.waitKey(1000)  # Wait for camera adjustments
        ret, image = cap.read()
        if not ret:
            print("ERROR: Could not read frame from camera.")
            return
        cv2.imwrite(calibration_image, image)
        cap.release()

    squares = find_squares(calibration_image, CALIBRATION_NTH_SQUARE)
    if not squares:
        print("ERROR: Calibration failed - no squares detected. Check captured image: " + calibration_image)
        return

    # Choose the Nth largest square (0 = largest, -1 = smallest)
    print(f'CALIBRATION: Using n-th square for calibration: {CALIBRATION_NTH_SQUARE} of {len(squares)} detected squares.')
    print("CALIBRATION: Check captured image: " + calibration_image)
    chosen_square = squares[CALIBRATION_NTH_SQUARE]
    calibration['coordinates'] = chosen_square.tolist()
    print(calibration['coordinates'])
    with open(CALIBRATION_FILE, 'w') as f:
        import json
        json.dump(calibration, f, indent=4)
    


if __name__ == '__main__':
    import os
    import json

    if os.path.isfile(CLASSES_FILE):
        try:
            with open(CLASSES_FILE, 'r') as f:
                config = json.load(f)
        except:
            print("ERROR: Problem loading evaluation data: " + CLASSES_FILE)

    if os.path.isfile(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                calibration = json.load(f)
        except:
            print("ERROR: Problem loading calibration data: " + CALIBRATION_FILE)

    if len(sys.argv) > 1 and sys.argv[1] == '--calibrate':
        calibration_image = None
        if len(sys.argv) > 2:
            calibration_image = sys.argv[2]
        calibrate(calibration_image)
    elif len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] == '--capture'):
        filenames = []
        while True:
            # Wait for key
            key = input("Press Enter to capture image from camera...").strip()
            if key.lower() == 'q':
                break
            elif key.lower() == 'c':
                calibrate()
                continue
            evaluate(filenames)
    else:
        filenames = sys.argv[1:]
        evaluate(filenames)
