import os
from unittest import result
import cv2
import utils
import numpy as np
import matplotlib.pyplot as plt


def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:A
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
    1. Read the txt file.
    2. Use cv2.imread() and os.path.join() to open the image in the .txt file
    3. Read the following lines of the .txt file and get the position of a face.
    4. Change the image to a gray scale image.
    5. Resize the image to the size of 19x19. 
    6. Use clf.classify() to detect faces.
       If it is face, the color of the box will be green, otherwise it will be red.
    """
    with open(dataPath, 'r') as file:
      lines = file.readlines()

    times = 0

    for line in lines:
      if(times == 0):
        img_name, number_of_face = line.split()
        image = cv2.imread(os.path.join("data/detect", img_name))
      else:
        left, upper, width, height = map(int, line.split())
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if clf.classify(cv2.resize(gray_image[upper:upper+height, left:left+width], (19, 19))):
          cv2.rectangle(image, (left, upper), (left+width, upper+height), (0, 255, 0), 3)
        else:
          cv2.rectangle(image, (left, upper), (left+width, upper+height), (0, 0, 255), 3)
      times = times + 1
      if(times == int(number_of_face)+ 1):
        times = 0
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # End your code (Part 4)
    
