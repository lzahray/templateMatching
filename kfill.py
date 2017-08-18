import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf 
import json
from PIL import Image
import pytesseract

def kfill(im, k, numPasses):
	assert k >= 5 
	image = im.copy()
	#dumb python method that will be slow. I hate python for loops :(
	#i give up we're killing our beautiful grayscale we once knew and loved. 
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	image = cv2.bitwise_not(image)
	cv2.imshow('original image', image)
	cv2.waitKey(0)
	for i in range(numPasses):
		print "next"
		cv2.imshow('image', image)
		cv2.waitKey(0)
		#cv2.waitKey(0)
		image = passOver(image, k, (i +1)% 2)
		
	return image

def passOver(im, k, high): #high is True if we're turning things text colored, False if we're turning things background colored
	#the annoying part about rotated image is it's no longer just black or white. I guess we'll define a thresh?
	image = im.copy()

	if not high:
		image = cv2.bitwise_not(image)
	anotherCopy = image.copy()
	# cv2.imshow('image', image)
	# cv2.waitKey(0)
	for i in range(image.shape[0]-k):
		for j in range(image.shape[1]-k):
			area = image[i:i+k, j:j+k].copy()
			#print "area is ", area
			center = area[2:k-2, 2:k-2]
			r = (int(area[0,0]) + int(area[0,k-1]) + int(area[k-1,0]) + int(area[k-1,k-1])) / 255 #corners
			n = (np.sum(area) - np.sum(center)) / 255
			#now when solving for c we first set the center to all white (we're not counting connections here)
			center[:,:] = 0
			c = cv2.connectedComponents(area)[0]-1
			if c == 1:
				if n > 3 * k - k/3.0 or (r==2 and n == 3*(k - k/3.0)):
				#if n > 3 * k +3 or (r==2 and n == 3*k + 3):
					#print "about to do a thing! Ish"
					area = anotherCopy[i:i+k, j:j+k] #just getting an actual version of area that's not a copy
					#print "area is "
					#print area
					center = area[2:k-2, 2:k-2]
					center[:,:] = 255 #paint it all black! woooooo
					#print "new area is "
					#print area
	#print "image shape is ", image.shape
	if high:
		return anotherCopy
	else:
		return cv2.bitwise_not(anotherCopy)

def test(image):
	image = cv2.resize(image, None, fx=0.5,fy=0.5,interpolation = cv2.INTER_AREA)
	newImage = kfill(image, 5, 4)
	cv2.imshow('newimage',newImage)
	cv2.waitKey(0)

imageForTest = 'forDemo/moreBadLines.tiff'
test(cv2.imread(imageForTest))




