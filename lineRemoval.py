import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf 
import json
from PIL import Image
import pytesseract
import argparse
import kfill


#alright lisa gots to write this as a function. Also at the end of today please make a commit. 

# imagesWithLineNoise = ['../myNewFaxes/590a05f991b5c93273777af2', '../myNewFaxes/590b65843839be54ebf452ef', '..myNewFaxes/591a0f2d91b5c91b2421d399', '../myNewFaxes/591b07c391b5c96ee3828ce2', '../myNewFaxes/591b08413839be646bd9dca1', '../myNewFaxes/591cbafc91b5c909590de115']


def removeLines(thresh, amountRatio=.55, minLineLengthRatio = 0.93, maxLineGap = 10): #so we don't actually care much about the amount ratio. Like imaging the case where every other pixel is black - we want to delete that line. I think
	#we're figuring out orientation and things after this function so we should indeed rotate if the dimensions are wrong page rotation
	#the plan: We do 2 different hough transforms, with a different threshold. 
	#If the line falls within a certain theta it's allowed to have fewer votes 
	#the page is a rectangle with different width and height

	#thresh should already be inverted broski
	thresh = thresh.copy() #so we don't modify the original
	#first make sure it's in expected page orientation (this doesn't take care of squished forms well or upside down-ness)
	#print "shape is ", thresh.shape
	if thresh.shape[1] > thresh.shape[0]:
		thresh = np.rot90(thresh)
		#image = np.rot90(image)
		#image = image.copy()
		thresh = thresh.copy()
		#print "we rotated"
	imageCopy = cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB)
	#cv2.imshow('original', image)
	#cv2.waitKey(0)

	#print "image shape after rot is ", image.shape
	#we're looking for lines that go basically all the way across
	linesLong = cv2.HoughLinesP(thresh, 1, np.pi/(360.), int(thresh.shape[0]*amountRatio), minLineLength = int(thresh.shape[0]*minLineLengthRatio), maxLineGap = maxLineGap)
	linesShort = cv2.HoughLinesP(thresh, 1, np.pi/(360.), int(thresh.shape[1]*amountRatio), minLineLength = int(thresh.shape[1]*minLineLengthRatio), maxLineGap = maxLineGap)


	#note to self: Test this more thoroughly to make sure we didn't done goof on the trig
	#print "long", linesLong
	#print "short", linesShort
	finalLines = []
	if linesLong is not None:
		for line in linesLong:
			for x1, y1, x2, y2 in line: #for some reason there's an extra layer within this list. I don't know why, I'm just a hairy guy. Don't ask me why **don't know**
				#since this is a long line we're happy no matter what angle it is :)
				cv2.line(thresh,(x1,y1),(x2,y2),0,1)
				cv2.line(imageCopy,(x1,y1),(x2,y2),(0,0,0),1)
				finalLines.append((x1,y1,x2,y2))
	if linesShort is not None:
		for line in linesShort:
			for x1,y1,x2,y2 in line: 
				#since this is a short line we have to constrain the allowed angle on this line
				# try:
				angle = np.arctan(float(y2-y1) / (x2-x1))
				#print "angle is ", angle
				# except ZeroDivisionError:
				# 	#this is apparently not working lol numpy is too smart! 
				# 	print "divide by zero"
					#angle = np.pi/2.0
				if abs(angle) < np.arccos(thresh.shape[1]/thresh.shape[0]): #if we're horizontal enough on the document 
					cv2.line(thresh,(x1,y1),(x2,y2),0,1)
					cv2.line(imageCopy,(x1,y1),(x2,y2),(255,0,0),1)
					finalLines.append((x1,y1,x2,y2))
				# else:
				# 	#print "the following line was a nogo: ", x1, y1, x2, y2
				# 	cv2.line(thresh,(x1,y1),(x2,y2),0,1)
	#print finalLines
	# cv2.imshow('colorful',imageCopy)
	# cv2.waitKey(0)
	linesFound = False
	if len(finalLines) > 0:
		linesFound = True
	return thresh, linesFound
	#cv2.imshow('houghlines', image)
	#cv2.waitKey(0)

	#now we're going to try removing the pixels that are on those lines. Follow the yellow brick line. Follow the follow the follow the follow the follow the yellow brick line
	
	#you guys i derped. I wrote this nice beautiful code for drawing a line, and it took an hour, and it already exists! I'm so sad! But my code is so pretty. Sigh I will replace it
	#but i'm keeping it commented out as a memory of what once was. Sob. 
	#apparently what I wrote is super close to this thing called the Bresenham line algorithm. If I'd been alive back in the 60s an hour of work would have gotten me a publication! Rough times we live in. Rough times indeed. 
	# for line in finalLines:
	# 	x1,y1,x2,y2 = line
	# 	cv2.line(imageCopy, (x1,y1), (x2,y2), (255,255,255),1)
	# 	cv2.line(thresh, (x1,y1), (x2,y2), 0,1)
		#rest in peace my beautiful line code. I will miss you. But this code is faster. I imagine. Mostly just because no python loops it's doing the same thing
		#man writing opencv seems like it would have been so fun i'd love that. Noise reduction is my new favorite

		# x1,y1,x2,y2 = line
		# markedPixels = set({(x1,y1)})
		# xbounds = np.sort([x1,x2])
		# ybounds = np.sort([y1,y2])
		# try:
		# 	m = float(y2-y1) / float(x2-x1)
		# 	b = y2-m*x2
		# 	end = (x1,y1)
		# 	print "first point is ", end

		# 	print "xbounds", xbounds
		# 	print "ybounds", ybounds
		# 	while end != (x2,y2): #the end will be nigh. It must be. Math. SCIENCE!!!
		# 		closestValue = np.inf #we're gonna want to get to 0
		# 		closestPixel = None
		# 		endx,endy = end
		# 		for x,y in [(endx+i,endy+j) for i in (-1,0,1) for j in (-1,0,1)]: 
		# 			if (i != 0 or j != 0) and (x,y) not in markedPixels: #we're checking the adjacent squares we haven't already visited
		# 				if (xbounds[0] <= x <= xbounds[-1]) and (ybounds[0] <= y <= ybounds[-1]):
		# 					answer = m*x+b-y
		# 					if abs(answer) < closestValue:
		# 						closestValue = abs(answer)
		# 						closestPixel = (x,y)
		# 		#print "closest pixel is ", closestPixel
		# 		markedPixels.update({closestPixel})
		# 		end = closestPixel
		# 				#now remember x,y is not what numpy thinks. Numpy is rows then columns. 
		# 	#I'm sorry who decided on this matrix notation that doesn't match x,y standards this is completely ridiculous I am angry. Anger!
		# 	indices = zip(*markedPixels)
		# 	if not flipped:
		# 		indices = [indices[1], indices[0]] #reorder for the numpyness. Strangely enough this is for the normal case of not flipping the x's and y's
		# 	#time to turn our image white! It's time the time has arrived whoop di doodle
		# 	imageCopy[indices] = 255
		# except ZeroDivisionError:
		# 	#this is an easy case! We have a beautiful vertical line! xs are equal
		# 	imageCopy[ybounds[0]:ybounds[-1], x1] = 255


	#cv2.imwrite("whynotaline.tiff",imageCopy)

	# cv2.imshow('whitedOut',thresh)
	# cv2.imshow('whited out on image', imageCopy)
	# cv2.waitKey(0)
	#fill.test(thresh, 1, True)


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help="path to input image file")
# ap.add_argument("-f", "--folder", help="path to folder containing many images")
# args = vars(ap.parse_args())
# images = []
# if args["image"] == args["folder"]:
# 	ap.error("Need an image (-i) xor a folder (-f) argument")
# elif args["image"]:
# 	images.append(cv2.imread(args["image"])) 
# else:
# 	fileNames = listdir(args["folder"])
# 	for f in fileNames:
# 		f = args["folder"] + '/' + f #get the actual location
# 		images.append(cv2.imread(f))

# for image in images:
# 	image = cv2.resize(image,None, fx=0.75, fy = 0.75, interpolation = cv2.INTER_AREA)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	gray = cv2.bitwise_not(gray)
# 	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 	newImage = removeLines(thresh)
# 	cv2.imshow('newImage', newImage)
# 	cv2.imshow('oldImage', thresh)
# 	cv2.waitKey(0)



