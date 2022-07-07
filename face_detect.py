import sys
import cv2 as cv

def main():
    imagePath = sys.argv[1]
    cascPath = "face.xml"
    detect_face(imagePath, cascPath)

def detect_face(image, file):
    faceCascade = cv.CascadeClassifier(file)

    # Read the image
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Face detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv.cv2.CASCADE_SCALE_IMAGE
    )

    print("Found {0} faces!".format(len(faces)))

    # draws a rectangle around the faces detected
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("Faces found", img)
    cv.waitKey(0)

if __name__ == '__main__':
    main()
