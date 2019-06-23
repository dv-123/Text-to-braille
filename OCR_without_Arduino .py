# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:29:42 2019

@author: bhaik
"""

#using Pytesseract for predictions

import cv2
import imutils
import numpy as np
from skimage.filters import threshold_local
import pytesseract
from PIL import Image
import time
from docx import Document

pytesseract.pytesseract.tesseract_cmd = 'Path_to_saved_model/Tesseract-OCR/tesseract'

document = Document()

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped




cap = cv2.VideoCapture(0)

image_path = "Path_to_the_image\\123.jpg"


while True:
    
    # finding edges
    sec = time.time()
    ret, frame = cap.read()
    ratio = frame.shape[0]/500.0
    orig = frame.copy()
 
    frame = imutils.resize(frame, height = 500)
    frame = cv2.bilateralFilter(frame,10,10,6)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray,10,10,6)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    # finding the contours
    
    _, cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    
    # loop over the contour
    for c in cnts:
        
        count=0
        count = count + 1
        
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        
        # if our approximated contour have 4 points then we assume that we have founded our screen
        
        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(frame, [screenCnt], -1, (0,255,0), 3)
            
            # apply a prespective Transform & Threshold
            wrapped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            wrapped = cv2.cvtColor(wrapped, cv2.COLOR_BGR2GRAY)
            T = threshold_local(wrapped, 11, offset = 10, method = "gaussian")
            wrapped = (wrapped > T).astype("uint8")*255
            
            cv2.imshow("scanned", imutils.resize(wrapped, height = 650))
            #time.sleep(1)
            #if count == 10000:
            #    cv2.imwrite(image_path, wrapped)
            #    break
    
    cv2.imshow("frame", frame)
    
    key = cv2.waitKey(1)
    if key == 27:
        cv2.imwrite(image_path, wrapped)
        break

cap.release()    
cv2.destroyAllWindows()

filename = "C:\\Users\\bhaik\\OneDrive\\Desktop\\Vibhav_projects\\saved_image\\123.jpg" 

def get_string(img_path):
    # Read image with opencv
    img = cv2.imread(img_path)

    # Convert to gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # filtering the image
    
    img = cv2.bilateralFilter(img,10,10,6)

    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)

    # Write image after removed noise
    cv2.imwrite("removed_noise.png", img)

    #  Apply threshold to get image with only black and white

    # Write the image after apply opencv to do some ...
    cv2.imwrite(img_path, img)
    # Recognize text with tesseract for python
    result = pytesseract.image_to_string(Image.open(img_path))
    # Remove template file
    #os.remove(temp)
    
    # print(type(result))
    
    #for i in range(len(result)):
    #    print(result[i])

    return result


print ('--- Start recognize text from image ---')
print (get_string(filename))

paragraph = document.add_paragraph(get_string(filename))
document.save('new-file-name.docx')
print ("------ Done -------")
