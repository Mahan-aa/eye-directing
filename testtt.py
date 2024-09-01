import cv2
import dlib
import numpy as np
from imutils import face_utils
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# Constants for the eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.3

# Initialize dlib's face detector and create the facial landmark predictor

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Grab the indexes of the facial landmarks for the left and right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize Tkinter
root = tk.Tk()
root.title("Eye Open/Closed Detection")
root.geometry("700x500")

# Create a label for the video feed
video_label = Label(root)
video_label.pack()

# Create a label for the status
status_label = Label(root, text="Status: ", font=("Helvetica", 16))
status_label.pack()


CONSEC_FRAMES = 3 
closed_eye_frames = 0
eye_closed = False


# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to update the video feed
def update_frame():
    global closed_eye_frames,eye_closed
    ret, frame = cap.read()
    if not ret:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            closed_eye_frames += 1
            if closed_eye_frames >= CONSEC_FRAMES: # make the program ignore blinking
                if not eye_closed:
                    eye_closed = True
                    status_label.config(text="Status: Eye Closed", fg="red")  # Red
        else:
            if eye_closed:
                eye_closed = False
                status_label.config(text="Status: Eye Open", fg="green")  # Green
            closed_eye_frames = 0

        # Update the status label


    # Convert frame to RGB and display in the Tkinter window
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Call the update_frame function after 10 milliseconds
    video_label.after(10, update_frame)

# Function to start the video stream
def start_video():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()

# Function to stop the video stream
def stop_video():
    cap.release()
    video_label.config(image='')

# Start and Stop buttons
start_button = Button(root, text="Start", command=start_video)
start_button.pack(side=tk.LEFT, padx=10, pady=10)

stop_button = Button(root, text="Stop", command=stop_video)
stop_button.pack(side=tk.RIGHT, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
