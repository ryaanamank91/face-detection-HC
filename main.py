import cv2
from functions import initialize_cam, load_haar_cascade, detect_faces, draw_rectangle, release_resources


def main():
    # Initialize the camera
    cap = initialize_cam()

    # Load the Haar Cascade classifier
    face_cascade = load_haar_cascade()

    while True:
        # Capture the frame from the camera
        success, frame = cap.read()

        if not success:
            print("Failed to grab the frame")
            break

        # Detect faces in the frame
        faces = detect_faces(face_cascade, frame)

        # Draw the rectangles around the detected faces
        draw_rectangle(frame, faces)

        # Display the frame with the detected faces
        cv2.imshow('Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and resources
    release_resources(cap)

if __name__ == '__main__':
    main()