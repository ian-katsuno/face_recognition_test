import face_recognition
import cv2
import numpy
import os

_dir = os.path.dirname(os.path.abspath(__file__))

inputPhotoDir = _dir + '/photos/input/'
outputPhotoDir = _dir + '/photos/output/'
frame = 'pureum.jpg'
photo = 'ian-hawaii.jpg'
staff = []

def process_staff_profile_pic(photo):
  image = face_recognition.load_image_file(photo)
  face_locations = face_recognition.face_locations(image)

  print("found {} faces in {}".format(len(face_locations), photo))

  # save an output image with blue box drawn around face
  for location in face_locations:
    saveOutputImage(image, location)
  
  encodings = face_recognition.face_encodings(image, face_locations)

  for encoding in encodings:
    print(encoding)
    staff.append(encoding)

  
def convert_css_to_coords(css_location):
  # css location is in the format top, right, bottom, left
  return ( (css_location[3], css_location[0]), (css_location[1], css_location[2]) )

def convert_PIL_to_opencv(pil_image):
  return cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)

def drawSquare(image, topLeft, bottomRight, color, thickness):
  # Using cv2.rectangle() method 
  # Draw a rectangle with blue line borders of thickness of 2 px 
  image = cv2.rectangle(image, topLeft, bottomRight, color, thickness) 
  return image

def saveOutputImage(image, location):
  print(location)
  topLeft, bottomRight = convert_css_to_coords(location)
  cvImage = convert_PIL_to_opencv(image)
  cvImage = drawSquare(cvImage, topLeft, bottomRight, (255, 0, 0), 2)
  cv2.imshow('display', cvImage)
  cv2.imwrite(outputPhotoDir + 'output2.jpg', cvImage)
  #cv2.waitKey(0)

def checkFrameForStaff(frame, staff):
  image = face_recognition.load_image_file(frame)
  face_locations = face_recognition.face_locations(image)

  print("found {} faces in {}".format(len(face_locations), frame))

  encodings = face_recognition.face_encodings(image, face_locations)

  for member in staff:
    matches = face_recognition.compare_faces(encodings, member)
    print(matches)

process_staff_profile_pic(inputPhotoDir + photo)
checkFrameForStaff(inputPhotoDir + frame, staff)

