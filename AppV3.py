import face_recognition
import cv2
import time
from concurrent.futures import ThreadPoolExecutor
import gc
gc.collect()
# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []
recognized_name = ""
recognized_time = 0
DISPLAY_TIME = 3  # seconds to display the recognized face

# Open the webcam
video_capture = cv2.VideoCapture(0)

# Set the frame skipping rate
frame_skipping_rate = 2
frame_count = 0

def process_frame(frame):
    global face_locations, face_encodings, recognized_name, recognized_time
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    name = "uncool"
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        if True in matches:
            name = "Langer Penis"
            recognized_name = name
            recognized_time = time.time()  # Record the time the face was recognized

with ThreadPoolExecutor(max_workers=2) as executor:
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        if not ret:
            break
        if frame_count % frame_skipping_rate == 0:
            # Process frame in a separate thread to prevent blocking the main loop
            executor.submit(process_frame, frame)
        frame_count += 1
        # Display the results
        for (top, right, bottom, left) in face_locations:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face if the face was recognized within the last DISPLAY_TIME seconds
            if recognized_name and (time.time() - recognized_time) < DISPLAY_TIME:
                name = recognized_name
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Display the resulting image
        cv2.imshow('Video', frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
