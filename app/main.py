from flask import Flask, render_template, Response
import copy
import cv2
import os
import mediapipe as mp
import time
import imageio
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
app = Flask(__name__)

gif_A = "Follow.gif"
gif_B = "Stop.gif"

# verbose flag will show overlay on webcam footage
VERBOSE = True

DURATION = 1

# Threshold for setting assured recognized gesture
threshold_gestures = 14  # 70% of 20

def gen():
    # Open the webcam (use 0 for the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    gif_start_time = time.time()  # Initialize gif_start_time outside the loop
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    try:
        # Load the gesture recognition model
        current_dir = os.getcwd()
        model_path = os.path.join(current_dir, 'gesture_recognizer.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.GestureRecognizerOptions(base_options=base_options)
        recognizer = vision.GestureRecognizer.create_from_options(options)

    except Exception as e:
        print(f"Error loading the model: {e}")
        exit()

    assured_recognized_gesture = None
    gif_window_open = False
    gif_frames = None
    gesture_stack = []
    pause_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        gesture_results = recognizer.recognize(mp_image)

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

                if pause_time and time.time() - pause_time < DURATION:
                    if assured_recognized_gesture == "Pointing_Up" and gif_A:
                        if not gif_window_open:
                            gif_frames = imageio.get_reader(gif_A)
                            gif_window_open = True

                        try:
                            gif_frame = gif_frames.get_next_data()
                        except Exception as e:
                            gif_window_open = False
                            gif_frames = None

                    elif assured_recognized_gesture == "Open_Palm" and gif_B:
                        if not gif_window_open:
                            gif_frames = imageio.get_reader(gif_B)
                            gif_window_open = True

                        try:
                            gif_frame = gif_frames.get_next_data()
                        except Exception as e:
                            gif_window_open = False
                            gif_frames = None

            annotated_image = frame.copy()

            if VERBOSE:
                # Draw hand landmarks
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

                # Display recognized gesture
                if assured_recognized_gesture:
                    cv2.rectangle(annotated_image, (0, 0), (400, 50), (0, 0, 0), -1)  # Black background
                    cv2.putText(annotated_image, f"Recognized: {assured_recognized_gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)  # White text

            frame = annotated_image
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            if assured_recognized_gesture and time.time() - pause_time < DURATION:
                if assured_recognized_gesture == "Pointing_Up" and gif_A:
                    if not gif_window_open:
                        gif_frames = imageio.get_reader(gif_A)
                        gif_window_open = True
                        gif_start_time = time.time()
                        gesture_stack = []


                elif assured_recognized_gesture == "Open_Palm" and gif_B:
                    if not gif_window_open:
                        gif_frames = imageio.get_reader(gif_B)
                        gif_window_open = True
                        gif_start_time = time.time()
                        gesture_stack = []


                if gif_window_open and gif_frames is not None:
                    elapsed_time = time.time() - gif_start_time
                    try:
                        fps = gif_frames.get_meta_data().get('fps', 15)  # Default to 30 if fps metadata is not available
                        frame_count = int(elapsed_time * fps)
                        gif_frame = gif_frames.get_data(frame_count % gif_frames.get_length())
                        ret, jpeg = cv2.imencode('.jpg', gif_frame)
                        gif_frame = jpeg.tobytes()
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + gif_frame + b'\r\n')
                    except Exception as e:
                        gif_window_open = False
                        gif_frames = None
            else:
                gif_window_open = False
                pause_time = None
                gif_frames = None
                pause_time = None
                assured_recognized_gesture = None

        if not gif_window_open:
                frame = annotated_image  # Show annotated image from webcam
                ret, jpeg = cv2.imencode('.jpg', frame)
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
