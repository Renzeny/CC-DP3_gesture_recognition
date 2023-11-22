import copy
import cv2
import os
import sys
import mediapipe as mp
import time
import imageio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

app_path = os.path.dirname(sys.executable)

# Helper function for visualizing gesture recognition results
def visualize_results(frame, gesture_results, verbose):
    mp_drawing = mp.solutions.drawing_utils

    if gesture_results:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated_image = copy.deepcopy(frame_rgb)

        if verbose:
            for hand_landmarks in gesture_results.hand_landmarks:
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
                ])

                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())

            if assured_recognized_gesture:
                cv2.rectangle(annotated_image, (0, 0), (400, 50), (0, 0, 0), -1)  # Black background
                cv2.putText(annotated_image, f"Recognized: {assured_recognized_gesture}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # White text

        cv2.imshow("Webcam Feed", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

gif_A = os.path.join(app_path, "Follow.gif")
gif_B = os.path.join(app_path, "Stop.gif")

# verbose flag will show overlay on webcam footage
VERBOSE = True

# duration of the gif & timeout to pause hand gesture recognition
DURATION = 2

# Initialize MediaPipe for gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load the gesture recognition model
try:
    # Load the gesture recognition model
    # Get the current working directory
    current_dir = os.getcwd()
    model_path = os.path.join(app_path, 'gesture_recognizer.task')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Open the webcam (use 0 for the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Create a window to display the camera feed
cv2.namedWindow("Webcam Feed", cv2.WINDOW_NORMAL)

# Flag to control the pause state and determine which gesture was recognized
pause_time = None
recognized_gesture = None
assured_recognized_gesture = None
gif_window_open = False
gif_reader = None
gif_frames = []

# Storing recognized gestures in a stack
gesture_stack = []

# Threshold for setting assured recognized gesture
threshold_gestures = 14  # 70% of 20

while True:
    timestamp_ms = int(time.time() * 1000)  # Get timestamp in milliseconds

    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Check if it's time to pause analysis
    if pause_time and time.time() - pause_time < DURATION:
        # Display a specific GIF based on recognized gesture
        if assured_recognized_gesture == "Pointing_Up" and gif_A:  # Checking if the GIF path is not empty
            if not gif_window_open:
                gif_frames = imageio.get_reader(gif_A)
                gif_window_open = True

            try:
                gif_frame = gif_frames.get_next_data()
                cv2.imshow("GIF Window", cv2.cvtColor(gif_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                gif_window_open = False
                gif_frames = None

        elif assured_recognized_gesture == "Open_Palm" and gif_B:  # Checking if the GIF path is not empty
            if not gif_window_open:
                gif_frames = imageio.get_reader(gif_B)
                gif_window_open = True

            try:
                gif_frame = gif_frames.get_next_data()
                cv2.imshow("GIF Window", cv2.cvtColor(gif_frame, cv2.COLOR_RGB2BGR))
            except Exception as e:
                gif_window_open = False
                gif_frames = None

    # Close GIF window after duration
    if gif_window_open and time.time() - pause_time >= DURATION:
        cv2.destroyWindow("GIF Window")
        gif_window_open = False
        gif_frames = None
        pause_time = None
        gesture_stack = []
        assured_recognized_gesture = None

        


    # Convert the frame received from OpenCV to a MediaPipe’s Image object.
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Perform gesture recognition on the frame
    gesture_results = recognizer.recognize(mp_image)

    # Check for specific gestures to pause
    if gesture_results:
        if gesture_results.gestures != []:
            recognized_gesture = gesture_results.gestures[0][0].category_name

            # Add recognized gesture to the stack
            gesture_stack.append(recognized_gesture)

            # Maintain the stack size to 20 elements
            if len(gesture_stack) > 20:
                gesture_stack.pop(0)
            
            if len(gesture_stack) == 20:
                # Check for the majority recognized gesture
                gesture_counts = {gesture: gesture_stack.count(gesture) for gesture in gesture_stack}
                max_count = max(gesture_counts.values())
                majority_gesture = [gesture for gesture, count in gesture_counts.items() if count == max_count]

                # Check if the majority gesture exceeds the threshold
                if (max_count / 20) * 100 >= threshold_gestures:
                    assured_recognized_gesture = majority_gesture[0]
                    pause_time = time.time()

    # Display the frame with gesture overlay in the window
    visualize_results(frame, gesture_results, VERBOSE)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Release the webcam
        cap.release() 
        # Exit all windows
        cv2.destroyAllWindows()
        break




