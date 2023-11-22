import copy
import cv2
import os
import mediapipe as mp
import time
import imageio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from flask import Flask, render_template, Response
import numpy as np

app = Flask(__name__)

gif_A = "Follow.gif"
gif_B = "Stop.gif"
threshold_gestures = 14 # 70% of 20
VERBOSE = True
DURATION = 2

# Initializing MediaPipe for gesture recognition
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load the gesture recognition model
try:
    current_dir = os.getcwd()
    model_path = 'gesture_recognizer.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)

except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame')

def visualize_results(frame, gesture_results, verbose):
    if frame is None or not isinstance(frame, np.ndarray):
        # Handle if the frame is empty or not a valid image
        return None

    mp_drawing = mp.solutions.drawing_utils
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    annotated_image = copy.deepcopy(frame_rgb)

    if gesture_results:

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

        # Convert annotated_image back to BGR format for OpenCV compatibility
        bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', bgr_image)
        frame_bytes = buffer.tobytes()

        return frame_bytes

def gen(cap):
    global assured_recognized_gesture  # Define it as a global variable
    pause_time = None
    gif_window_open = False
    assured_recognized_gesture = None  # Initialize the variable here
    gesture_stack = []  # Initialize gesture_stack here

    while True:

        try: 

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
                        frame = gif_frame  # Replace frame with GIF frame
                    except Exception as e:
                        gif_window_open = False
                        gif_frames = None

                elif assured_recognized_gesture == "Open_Palm" and gif_B:  # Checking if the GIF path is not empty
                    if not gif_window_open:
                        gif_frames = imageio.get_reader(gif_B)
                        gif_window_open = True

                    try:
                        gif_frame = gif_frames.get_next_data()
                        frame = gif_frame  # Replace frame with GIF frame
                    except Exception as e:
                        gif_window_open = False
                        gif_frames = None

            # Close GIF window after duration
            if gif_window_open and time.time() - pause_time >= DURATION:
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


            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            annotated_frame_bytes = visualize_results(frame, gesture_results, VERBOSE)
            if annotated_frame_bytes:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + annotated_frame_bytes + b'\r\n')
                
        except Exception as e:
            print(f"Error in frame generation: {e}")
            continue  # Continue with the next iteration

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
