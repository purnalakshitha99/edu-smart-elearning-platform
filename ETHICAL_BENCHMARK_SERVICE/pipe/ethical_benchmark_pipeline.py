import cv2
import mediapipe as mp
import numpy as np
import time
import torch
from flask import Flask
from db import mongo  # Import mongo instance
from bson import ObjectId
import winsound  # Import for beep sound (Windows only)

unauthorized_objects = ['cell phone', 'laptop', 'book', 'paper', 'calculator', "tablets"]
LOOKING_AWAY_THRESHOLD = 3

class EthicalBenchMarkDetector:
    def __init__(self, app, user_id):
        self.app = app
        self.model = self.load_model()
        self.user_id = user_id
        self.classes = self.model.names
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.unauthorized_objects = unauthorized_objects
        self.last_look_time = time.time()
        self.looking_away_threshold = 4
        self.detected_objects = []
        self.time_limit = 60

        self.current_looking_direction = "Looking forward"
        self.looking_direction_start_time = None
        self.looking_directions = {
            "Left": 0,
            "Right": 0,
            "Up": 0,
            "Down": 0
        }

        self.looking_threshold = 3  # Minimum time to look away before marking as potential cheating
        self.beep_sound_played = False  # Flag to prevent repeat beeps
        self.directions_beeping = {  # Track beeping status for each direction
            "Left": False,
            "Right": False,
            "Up": False,
            "Down": False
        }

    def plot_boxes(self, result, frame, looking):
        labels, cord = result
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.25:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
                label = self.class_to_label(labels[i])
                confidence = round(float(row[4]) * 100, 2)

                if label in self.unauthorized_objects:
                    text = f"{label}: {confidence}% (Unauthorized)"
                    self.detected_objects.append({
                        'object': label,
                        'confidence': confidence
                    })
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    text = f"{label}: {confidence}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(frame, looking, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame

    def score_frame(self, frame):
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def class_to_label(self, x):
        return self.classes[int(x)]

    def process_frame(self, image):
        img_h, img_w, img_c = image.shape
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        image_rgb.flags.writeable = True

        face_3d = []
        face_2d = []
        looking = "Looking forward"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])

                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])
                dis_matrix = np.zeros((4, 1), dtype=np.float64)

                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dis_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360

                looking_direction = ""
                if y < -5:
                    looking = "Looking Left"
                    looking_direction = "Left"
                elif y > 5:
                    looking = "Looking Right"
                    looking_direction = "Right"
                elif x < -5:
                    looking = "Looking Down"
                    looking_direction = "Down"
                elif x > 5:
                    looking = "Looking Up"
                    looking_direction = "Up"
                else:
                    looking = "Looking forward"
                    looking_direction = "Forward"

                current_time = time.time()

                if looking_direction:
                    if looking_direction in self.looking_directions:
                        if looking_direction == "Forward":
                            continue
                        direction_key = looking_direction

                        if self.looking_directions[direction_key] == 0:
                            self.looking_directions[direction_key] = current_time
                    else:
                        self.looking_directions = {
                            "Left": 0,
                            "Right": 0,
                            "Up": 0,
                            "Down": 0
                        }

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)

                cv2.line(image, p1, p2, (255, 0, 0), 3)

                results = self.score_frame(image)
                image = self.plot_boxes(results, image, looking)

        return image

    def run(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        time_limit = self.time_limit  # Store time_limit locally

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return

        try:
            while True:
                # Calculate elapsed time BEFORE reading the frame
                elapsed_time = time.time() - start_time

                # Check if time limit is exceeded BEFORE processing the frame
                if elapsed_time >= time_limit:
                    print("Time limit exceeded. Exiting.")
                    break  # Exit the loop

                success, image = cap.read()  # Read the frame
                if not success:
                    print("Ignoring empty camera frame.")
                    break

                processed_image = self.process_frame(image)
                cv2.imshow('Ethical Benchmark', processed_image)

                # Checking headpose direction
                for direction, start_time_dir in self.looking_directions.items():
                    if start_time_dir != 0:
                        duration = time.time() - start_time_dir
                        if direction != "Forward" and duration >= self.looking_threshold: # Excluding forward
                            print(f"Warning: Looking {direction} for {duration:.2f} seconds!")
                            self.detected_objects.append({
                                'event': f'Looking {direction} for extended period',
                                'duration': duration
                            })

                            # Check if beep has already been played for this direction
                            if not self.directions_beeping[direction]:
                                self.play_beep_sound()  # Play beep sound
                                self.directions_beeping[direction] = True  # Set beep flag for this direction

                            self.looking_directions[direction] = 0  # Reset direction time
                    elif self.directions_beeping[direction]:  # Reset the flag when not looking
                        self.directions_beeping[direction] = False


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

            # Save detected objects to MongoDB after processing is complete
            self.save_to_db(self.detected_objects, self.user_id, self.looking_directions)
            self.detected_objects = []

    def save_to_db(self, detected_objects, user_id, looking_directions):
        with self.app.app_context():
            cheating_data = {
                'timestamp': time.time(),
                'cheating_events': detected_objects,
                'user_id': ObjectId(user_id),
            }
            try:
                print("Attempting to save data to MongoDB...")
                mongo.db.cheating_events.insert_one(cheating_data)
                print("Cheating data saved to MongoDB.")
            except Exception as e:
                print(f"Error saving to MongoDB: {e}")

    def play_beep_sound(self):
         # Plays a beep sound (Windows only).  Other OS need different methods.
        frequency = 2500  # Set Frequency To 2500 Hertz
        duration = 1000  # Set Duration To 1000 ms == 1 second
        winsound.Beep(frequency, duration)