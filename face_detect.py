import sys 
import cv2 as cv

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "face.xml"

# Create the haar cascade
faceCascade = cv.CascadeClassifier(cascPath)

# Read the image
image = cv.imread(imagePath)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv.cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draws a rectangle around the faces
for (x, y, w, h) in faces:
    cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv.imshow("Faces found", image)
cv.waitKey(0)