import face_recognition
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

# Load a sample picture and learn how to recognize it.
known_image = face_recognition.load_image_file("known_person.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []
recognized_face_location = None
recognized_time = 0
DISPLAY_TIME = 3  # seconds to display the recognized face

# Open the webcam
video_capture = cv2.VideoCapture(0)

# Set the frame skipping rate
frame_skipping_rate = 2
frame_count = 0

def process_frame(frame):
    global face_locations, face_encodings, recognized_face_location, recognized_time
    # Resize frame of video to 1/2 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    found_match = False
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces([known_encoding], face_encoding)
        if True in matches:
            recognized_face_location = [coordinate * 2 for coordinate in face_location]  # Scale back up
            recognized_time = time.time()  # Record the time the face was recognized
            found_match = True
            break

    # If no match found, check the time since the last recognition
    if not found_match and (time.time() - recognized_time) > DISPLAY_TIME:
        recognized_face_location = None

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
        if recognized_face_location:
            top, right, bottom, left = recognized_face_location
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, "Langer Penis", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
