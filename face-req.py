#!/usr/bin/python

# Raspberry Pi Facial Recognition Setup Script
# Copyright, Tom Alessi, 2021

# Based on prior work done here:
# https://pimylifeup.com/raspberry-pi-opencv/
# https://www.pyimagesearch.com/2018/06/25/raspberry-pi-face-recognition/
# https://www.tomshardware.com/how-to/raspberry-pi-facial-recognition


import os
import time
import face_recognition
import pickle
import cv2
from imutils import paths, resize
from imutils.video import VideoStream
from imutils.video import FPS
from optparse import OptionParser


# Data directory (where to store photos and the serialized pickle file)
data_dir = '/home/pi/face-req-data'
# HAAR file (this should not change with a standard OpenCV install)
haar_file = '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'


def take_photo(name):
    # Take photos of a person and store them in a known place for training.
    
    # Create the dataset folder is not there
    if not os.path.exists(data_dir + '/' + name):
        # Create a directory in the data folder for this person
        print("Creating photo data directory.")
        os.makedirs(data_dir + '/' + name)
    
    # Setup the camera
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Press spacebar to take a photo.", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Press spacebar to take a photo.", 500, 300)

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab a frame.")
            break
        cv2.imshow("Press spacebar to take a photo.", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Esc hit, closing.")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = data_dir + "/" + name + "/image_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()
    cv2.destroyAllWindows()


def train():
    # Train the model from our photos
    
    print("Processing images.")
    imagePaths = list(paths.list_images(data_dir))

    # Initialize known encodings and names
    knownEncodings = []
    knownNames = []

    # Check all image paths
    for (i, imagePath) in enumerate(imagePaths):
        # Obtain the name of the person from the image path.
        print("Processing image {}/{}.".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # Load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,
            model="hog")

        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Loop over the encodings
        for encoding in encodings:
            # Add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    # Serialize the facial encodings and names to disk using pickle
    print("Serializing facial encodings.")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(data_dir + "/encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()


def analyze():

    # Initialize 'currentname' to trigger only when a new person is identified.
    currentname = "unknown"
    # Serialized pickle file with our known faces
    encodings_pickle = data_dir + "/encodings.pickle"

    # Load the known faces and OpenCV's Haar
    print("Loading serialized encodings.")
    data = pickle.loads(open(encodings_pickle, "rb").read())
    detector = cv2.CascadeClassifier(haar_file)

    # Initialize the video stream
    print("Starting video stream.")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    # Start the FPS counter
    fps = FPS().start()

    # Loop over frames from the video file stream
    while True:
        # Grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = resize(frame, width=500)
        
        # Convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # Compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown" #if face is not recognized, then print Unknown

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
                
                #If someone in your dataset is identified, print their name on the screen
                if currentname != name:
                    currentname = name
                    print(currentname)
            
            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # Draw the predicted face name on the image - color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (0, 255, 255), 2)

        # Display the image to our screen
        cv2.imshow("Facial recognition is running.", frame)
        key = cv2.waitKey(1) & 0xFF

        # Quit when 'q' key is pressed
        if key == ord("q"):
            break

        # Update the FPS counter
        fps.update()

    # Stop the timer and display FPS information
    fps.stop()
    print("Elasped time: {:.2f}".format(fps.elapsed()))
    print("Approximage FPS: {:.2f}".format(fps.fps()))

    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()

#
# Parse input arguments
#
parser = OptionParser(description='Raspberry Pi Facial Recognition', version='%prog 1.0')
parser.add_option("-n", "--name", dest="name", help="Name is person when taking photos")
parser.add_option("-c", "--command", dest="command", help="Command to run.  Options are: [photo|train|analyze|clean]")
(options, args) = parser.parse_args()

# Create the dataset folder is not there
if not os.path.exists(data_dir):
    # Creating data directory
    print("Creating data directory.")
    os.makedirs(data_dir)

if options.command == 'photo':
    print("Starting photo creator.")
    if options.name:
        take_photo(options.name)
    else:
        print("You must provide a name when taking photos.")
elif options.command == 'train':
    print("Training facial recognition model.")
    train()
elif options.command == 'analyze':
    print("Starting video analyzer.")
    analyze()
elif options.command == 'clean':
    print("Cleaning up data directory.")
    os.system('rm -rf %s/*' % data_dir)
else:
    print("Invalid or missing command")
    exit(2)


# All done
exit(0)
