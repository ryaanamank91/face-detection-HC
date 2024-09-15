import cv2


def initialize_cam():
    """Initialize the camera"""
    return cv2.VideoCapture(0)  # '0' is the default camera of the system, needs to changed accordingly


def load_haar_cascade():
    """Load the Haar Cascade model for face detection"""
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(face_cascade, frame):
    """Detect faces in the given frame"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Covert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces


def draw_rectangle(frame, faces):
    """Draw rectangles around detected faces"""
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def release_resources(cap):
    """Release the webcam and destroy OpenCV windows"""
    cap.release()
    cv2.destroyAllWindows()