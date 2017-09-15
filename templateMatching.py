import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from os import listdir
import tensorflow as tf 
import json
from PIL import Image
import pytesseract
from lineRemoval import removeLines
from subprocess import call 
import pandas as pd
#We're working with numpy standards. For us, x refers to rows and y refers to columns. Opencv is oposite but what can you do

#TODO Immediate: 
#Deal with multiple page tiffs, find the one probably with the qr code on it or something - there has to be an easy way
#Deal with scaling - the ones that are horizontal should be rescaled to be vertical (it's an approximation for now)
#Deal with upside-down, should be easy
#Write everything into nice functions, clean up code
#Write for multiple files, print out for each what boxes it thinks are checked

#TODO Less Immediate:
#Clean up documents, get rid of noise
#Figure out how to detect a check/X in the box - machine learning you can't run forever. Sometimes people write Os or just slashes, really we're looking for not noise and not long line - so I guess something connected but not too connected.
#Get rid of horrendus for loop, can probably use numpy smartness to make things faster

###############
#First we're going to set up this neural net for character recognition
#Idea for later: Actually take the subtraction for this area, because we'd like to get rid of the black border. 
#There has to be an easy way to then locate the number instead of just trying a bajillion things.... we could also just try a bajillion things
#Worried our training data has characters too close to the middle - maybe we should retrain translating and resizing for more data

#1 used to be none
x = tf.placeholder(tf.float32, [36,64]) #pixels of our input 28x28 everything is terrible maybe this will be ok

with open('numWithBoxesData.txt') as data_file:
    data = json.load(data_file)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') #ima guess W is our feature vector. Always make first and last 1 because why would we skip an image (batch) or... idk what the last is honestly but don't want to skip any

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1, 2, 2, 1], padding = 'SAME') 

W_conv1 = tf.reshape(data["W_conv1"], [8,8,1,32])
b_conv1 = tf.reshape(data["b_conv1"], [32])
W_conv2 = tf.reshape(data["W_conv2"], [5,5,32,64])
b_conv2 = tf.reshape(data["b_conv2"], [64])
W_conv3 = tf.reshape(data["W_conv3"], [3,3,64,64])
b_conv3 = tf.reshape(data["b_conv3"], [64])
W_fc1 = tf.reshape(data["W_fc1"], [5*8*64, 2048])
b_fc1 = tf.reshape(data["b_fc1"], [2048])
W_fc2 = tf.reshape(data["W_fc2"], [2048,13])
b_fc2 = tf.reshape(data["b_fc2"], [13])

#x = tf.placeholder(tf.float32, [None, 784]) #pixels for input
x_image = tf.reshape(x, [-1, 36, 64, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
h_pool3_flat = tf.reshape(h_pool3, [-1, 5*8*64]) #still confuzzled

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 #these are outputs not normalized yet

y_probabilities = tf.nn.softmax(y_conv) #I think this should make it probabilities, but it only makes sense if we feed in one image





######################################
np.set_printoptions(threshold=np.nan)
#from imutils, rotates without cutting off corners
def rotate_bound(image, angle): #angle in rad
    # grab the dimensions of the image and then determine the
    # center
    angle = angle * 180 / np.pi
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    warped = cv2.warpAffine(image, M, (nW, nH)) #alas this won't be BW anymore but it's pretty illegible if you make it BW so... rip
    return warped

def prepare(image): #this will resize to 0.5 at the end
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bitwise_not(gray)
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	#line noise removal
	thresh, foundALine = removeLines(thresh)
	#cv2.imshow('after line removal', thresh)

	#print thresh[50:200, 200:400]
	lines = cv2.HoughLines(thresh,1,np.pi/(360*3), int(thresh.shape[1]/2.2))
	thetaVotes = {}
	for line in lines:
		for rho,theta in line:
			if theta in thetaVotes:
				thetaVotes[theta] += 1
			else:
				thetaVotes[theta] = 0
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
	angle = np.pi / 2.0 - max(thetaVotes, key=thetaVotes.get) #- np.pi / 2.0
	#print "angle is ", angle
	rotatedImage = rotate_bound(thresh,angle)
	#cv2.imshow('after our rotation det',rotatedImage)
	#tesseract time, for some reason tesseract hates us and reasonable file extensions. We have to make a temporary image. It sucks
	#Do .05 and .95 to attempt to take out the upside-down text that sometimes appears. Idk if it matters but it def. doesn't hurt
	cv2.imwrite('temp.tiff', rotatedImage[int(rotatedImage.shape[0]*.05):int(rotatedImage.shape[0]*.95), int(rotatedImage.shape[1]*.05): int(rotatedImage.shape[1]*.95)])
	call(["tesseract", "temp.tiff", "../myTestOutput", "-l", "eng", "--psm", "0"])
	with open('../myTestOutput.osd') as myFile:
		output = myFile.readlines()
	#print output
	degrees = [int(s) for s in output[1].split() if s.isdigit()][0]
	#print "degrees is ", degrees
	if degrees: #there should be degrees but I'm not taking any chances! 
		rotatedImage = rotate_bound(rotatedImage,np.deg2rad(degrees))

	#cv2.imshow('after tess',rotatedImage)
	if rotatedImage.shape[1] > rotatedImage.shape[0]:
		#print "it needs to be scaled"
		rotatedImage = cv2.resize(rotatedImage, dsize=(int(1728), int(2131)), interpolation = cv2.INTER_NEAREST)

	#print "after scaling the size is ", finalImage.shape
	finalImage = cv2.resize(rotatedImage,None, fx=0.5, fy = 0.5, interpolation = cv2.INTER_AREA)
	#cv2.imshow('houghlines',image)
	#cv2.waitKey(0)
	return finalImage, foundALine
	#cv2.imshow("rotatedGray", rotatedImage)
	#cv2.waitKey(0)
	#cv2.imwrite("rotatedPage92.tiff", rotatedImage)
	

def templateMatch(template, mainImage, sizes = [0.85, 1.15], numSizesToTry = 20):
	minValue = np.inf
	minLoc = None
	myRatio = 0
	myHappyTemplate = template
	#allOfIt = None
	ratio = np.linspace(sizes[0],sizes[1],numSizesToTry)
	for i in range(numSizesToTry):
		#print i
		for j in range(numSizesToTry):
			if ratio[i] * template.shape[0] <= mainImage.shape[0] and ratio[j] * template.shape[1] <= mainImage.shape[1]: #if it will fit
				if ratio[i] < 1 and ratio[j] < 1: #I don't know if this is actually what we want
					newSizeTemplate = cv2.resize(template,None, fx=ratio[j], fy = ratio[i], interpolation = cv2.INTER_AREA)
				else:
					newSizeTemplate = cv2.resize(template,None, fx=ratio[j], fy = ratio[i], interpolation = cv2.INTER_CUBIC)
				result = cv2.matchTemplate(mainImage, newSizeTemplate, cv2.TM_SQDIFF_NORMED)
				minMaxLoc =  cv2.minMaxLoc(result) 
				if minMaxLoc[0] < minValue:
					minValue = minMaxLoc[0]
					minLoc = (minMaxLoc[2][1], minMaxLoc[2][0]) #THIS IS SO DUMB this function gives the reverse thing
					myRatio = (ratio[i], ratio[j])
					myHappyTemplate = newSizeTemplate.copy()
					#allOfIt = minMaxLoc
		# if i % 5 == 0:
		# 	cv2.imshow('step', findDarkValue(myRatio, minLoc,mainImage)[1])
		# 	cv2.waitKey(0)
	#print "ratio is ", myRatio
	#print "myRatio after tm is ", myRatio
	return minValue, minLoc, myRatio, myHappyTemplate				
	# print "min value ", minValue
	# print "min loc", minLoc #actually x,y order
	# print "my ratio is ", myRatio
	# print "all of it is ", allOfIt
	# cv2.imwrite("myHappyTemplate.tiff", myHappyTemplate)

def calculatePixels(key, myRatio, minLoc): #minLoc is coordinate where the template should sit, myRatio is the scaling of the template, key is the box we're looking at
	pixels = checkBoxPixels[key]
	#print "pixels are ", pixels
	#print "myRatio is ", myRatio
	pixels = np.array([pixels[0] * myRatio[0], pixels[1] * myRatio[0], pixels[2] * myRatio[1], pixels[3] * myRatio[1]]).astype(int) #this should give us the coordinates relative to top corner of resized template
	pixels = np.array([pixels[0] + minLoc[0], pixels[1] + minLoc[0], pixels[2] + minLoc[1], pixels[3] + minLoc[1]])
	#in the order xtop xbottom yleft yright
	return pixels


def findDarkValue(myRatio, minLoc, mainImage):
	color = cv2.cvtColor(mainImage,cv2.COLOR_GRAY2RGB) #just for drawing purposes
	#darkValue = {"30-Day Supply": 0, "90-Day Supply": 0, "Number of Additional Refills": 0, "Have Patient Contact My Office":0, "No Longer a Patient": 0, "Already Sent a Refill": 0, "Patient Unknown to this Office": 0, "Refill Too Soon": 0, "Other Reason for Denial": 0}
	darkValue = {}
	for key in checkBoxPixels:
		#if key != 'Number of Additional Refills':
		pixels = calculatePixels(key, myRatio, minLoc)
		#print mainImage[pixels[0]:pixels[1],pixels[2]:pixels[3]]
		darkValue[key] = int(np.sum(mainImage[pixels[0]:pixels[1],pixels[2]:pixels[3]])) #we're doing the int thing just for the json
		cv2.line(color, (pixels[2], pixels[0]), (pixels[3],pixels[0]), (0,0,255),2)
		cv2.line(color, (pixels[2],pixels[0]), (pixels[2],pixels[1]), (0,0,255),2)
		cv2.line(color, (pixels[2],pixels[1]), (pixels[3],pixels[1]), (0,0,255),2)
		cv2.line(color, (pixels[3],pixels[1]), (pixels[3],pixels[0]), (0,0,255),2)
	return darkValue, color

#def readNumber(myHappyTemplate, myRatio, minLoc, mainImage, color, hopSize): 
def readNumber(myRatio, minLoc, mainImage): 
	myPixels = calculatePixels("DigitPrediction", myRatio, minLoc)
	#xtop, xbottom, yleft, yright
	boxImage = mainImage[myPixels[0]:myPixels[1], myPixels[2]:myPixels[3]]
	boxImage = cv2.resize(boxImage, dsize = (64,36))
	# cv2.imshow('boxImage', boxImage)
	# cv2.waitKey(0)
	with tf.Session() as sess:
		yProb = sess.run(y_probabilities, feed_dict = {x: boxImage, keep_prob: 1.0}) #run on the slidy thing and on the center of mass thing. We'll end up taking the thing it's most confident of anyway
	return yProb



#input the pixels where we should be hunting
	#first before we get fancy let's just slide among these pixels. Then we'll get fancy. 
	#we're working in terms of a numpy array, so topLeft will be (row, column)
	# assert 0< hopSize <= 28
	# new = mainImage.copy()
	# new[minLoc[0]:minLoc[0]+myHappyTemplate.shape[0], minLoc[1]:minLoc[1]+myHappyTemplate.shape[1]] -= myHappyTemplate
	# #in the order xtopleft xbottomright ytopleft ybottomright
	# pixels = calculatePixels("Number of Additional Refills", myRatio, minLoc)
	# pixels[0] -= 4
	# pixels[1] += 4
	# pixels[2] -= 4
	# pixels[3] += 4 #make the box a bit bigger
	# height = pixels[1] - pixels[0] #this will be the length of our square as we slide forth and prosper
	# #we want to resize it to 28 in height, same aspect ratio
	# #print "height is ", height

	# resizeRatio = 28.0/height
	# #print "resize ratio is ", resizeRatio
	# rectangle = new[pixels[0]:pixels[1],pixels[2]:pixels[3]].copy()
	# #ok first we're gonna try subtracting it out. This will be difficult without my tongue
	# #rectangle = np.pad(rectangle, ((2,2), (0,0)), 'constant')
	# resizeRatio = 28.0 / height
	# #print "thing that's failing is ", int(resizeRatio * rectangle.shape[1])
	# rectangle = cv2.resize(rectangle, (int(resizeRatio * rectangle.shape[1]), 28)) #not gonna bother with interpolation here
	# width = rectangle.shape[1]
	# #print "rectangle size is ", rectangle.shape

	# #alternative plan awaits! 
	# #center of mass
	# #ok I just found out this is a thing that exists... rip me cv2.moments() 
	# columns = rectangle.sum(0) #sum down each column
	# rows = rectangle.sum(1) #sum across each row
	# colA = np.arange(columns.shape[0])
	# rowA = np.arange(rows.shape[0])

	# centerCol = int(np.sum(columns*colA) / float(np.sum(columns)))
	# centerRow = int(np.sum(rows*rowA) / float(np.sum(rows)))

	# squareToLookAt = new[ int(pixels[0] + (centerRow - 14) * resizeRatio): int(pixels[0] + resizeRatio * (centerRow + 14)), int(pixels[2] + resizeRatio * (centerCol - 14)): int(pixels[2] + resizeRatio * (centerCol + 14))]
	# squareToLookAt = cv2.resize(squareToLookAt, (28,28))
	# cv2.imshow("lastSquare", squareToLookAt)
	# #cv2.waitKey(0)

	# #####################
	# #Ok now what we want to do is pad our rectangle so that we can slide our box along some happy amount of times.
	# #hopSize time here we go
	# #n is number of windows we're gonna try fitting
	# n = int(np.ceil((width - 28) / float(hopSize) + 1))
	# #p is total amount of padding we need
	# p = hopSize * (n-1) + 28 - width
	# #print "p is ", p
	# #print "n is ", n
	# paddedRectangle = np.pad(rectangle, ((0,0), (p/2, p-p/2)), 'constant') #0 padding ftw
	# #cool beans now we're ready to roll lez doo diz
	# #we'll do a for loop for now, there are sooooo many ways we can make this code faster let me tell you my friend
	# myNewX = np.empty((n+1,28,28))
	# for i in range(n):
	# 	startLoc = hopSize * i
	# 	square = paddedRectangle[:,startLoc:startLoc+28]
	# 	cv2.imshow(str(i), square)
	# 	#now we wnat to run number detection on our happy square we worked hard for. Oh shoot I wonder if mnist.... white numbers or black numbers it matters... rats
	# 	#Alright for now let's assume the numbers are... well what we hope they are broski
	# 	myNewX[i,:,:] = square
	# myNewX[n,:,:] = squareToLookAt
	# #ok so now myNewX is what x should be in our tensorflow model. so we believe. so we believe folks i'm not exactly sure 
	# #print "myNewX.shape is ", myNewX.shape
	# with tf.Session() as sess:
	# 	yConv = sess.run(y_conv, feed_dict = {x: myNewX, keep_prob: 1.0}) #run on the slidy thing and on the center of mass thing. We'll end up taking the thing it's most confident of anyway

	# 	#also run on the COM thing
	# 	# altX = np.empty((1,28,28))
	# 	# altX[0,:,:] = squareToLookAt
	# 	# yOther = sess.run(y_probabilities, feed_dict = {x: altX, keep_prob: 1.0})
	# 	# yOtherC = sess.run(y_conv, feed_dict = {x: altX, keep_prob: 1.0})
	# # print "yother is ", yOther
	# # print "yOtherC is ", yOtherC
	# #print yConv
	# return yConv

########################################


def justDoNumbers():
	print "in the func"
	with open('checkboxAnalysis/turningDFToJson.txt') as data_file:
		allInfo = json.load(data_file)
	allInfo = pd.read_json(allInfo)
	correct = 0
	incorrect = 0
	skip = 0
	for imageFileName in images.keys():
		try:
			image = images[imageFileName]
			faxName = imageFileName.replace("../myNewFaxes/", '').replace(".tiff", '')
			row  = allInfo.loc[allInfo['fax_id']==faxName].iloc[0]
			minLoc = row['templateLocation']
			myRatio = row['templateRatio']
			
			#no need to do template match though
			if row['Number of Additional Refills'] > 5000 and row['document_type'] == 'Prescription':
				mainImage, foundALine = prepare(image)
				number = readNumber(myRatio, minLoc, mainImage)
				print "probs ", number
				maxNum = np.argmax(number)
				print "prediction: ", maxNum
				actual = row['initial_refills'] 
				print "actual: ", actual
				if actual == maxNum:
					print "CORRECT"
					correct += 1
				else:
					print "INCORRECT"
					incorrect += 1
			else:
				print "skip"
				skip += 1
			print "total correct: ", correct
			print "total incorrect: ", incorrect
			print "total skip: ", skip
			print '\n'
		except IndexError:
			print "index error! \n"

#in the order xtop xbottom yleft yright the way lisa thinks about x and y (think numpy)
checkBoxPixels = {"30-Day Supply": [45,61,23,40], "90-Day Supply": [72,90,23,40], "Number of Additional Refills": [100,118,23,66], "Have Patient Contact My Office":[45,63,391,409], "No Longer a Patient": [73,91,391,409], "Already Sent a Refill": [101,119,391,409], "Patient Unknown to this Office": [129,146,391,409], "Refill Too Soon":[156,174,391,409], "Other Reason for Denial": [199,218,391,737], "DigitPrediction":[95,131,14,78]}

# parse arguments, load images
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input image file")
ap.add_argument("-f", "--folder", help="path to folder containing many images")
args = vars(ap.parse_args())
#print args
#args = ap.parse_args()
images = {}
if args["image"] == args["folder"]:
	ap.error("Need an image (-i) xor a folder (-f) argument")
elif args["image"]:
	images[args["image"]] = cv2.imread(args["image"])
else:
	fileNames = listdir(args["folder"])
	for f in fileNames:
		f = args["folder"] + '/' + f #get the actual location
		images[f] = cv2.imread(f)

##########################################

#set up template that's been scaled by 0.5 already

template = cv2.imread('../rotatedTemplate.tiff') 
#print template
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.bitwise_not(template)
ret, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
template = template[380:621, 30:800] #We manually figured out these pixels

justDoNumbers()
####################################

#perform detection for each image



def main():
	listOfDicts = []
	imageNumber = 0
	for imageFileName in images.keys():
		try:
			print imageFileName
			print "image ", imageNumber
			information = {}

			image = images[imageFileName]

			information["imageName"] = imageFileName
			#cv2.imshow('original', image)
			mainImage, foundALine = prepare(image)
			information["foundALine"] = foundALine
			cv2.imshow('final', mainImage)
			cv2.waitKey(0)
			#break

			minValue, minLoc, myRatio, myHappyTemplate	= templateMatch(template, mainImage, numSizesToTry = 30)
			information["templateRatio"] = myRatio
			information["templateMatchValue"] = minValue
			information["templateLocation"] = minLoc

			if minLoc: #if the template matching actually worked

				darkValue, color = findDarkValue(myRatio, minLoc, mainImage)

				information.update(darkValue)
				#print darkValue
				#print "max is ", max(darkValue, key=darkValue.get)
				#print "\n"

				# ret,thresh = cv2.threshold(mainImage,127,255,0)
				# im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
				# print "contours ", contours
				# cv2.imshow("im2", im2)
				# cv2.waitKey()
				# cont = cv2.drawContours(cv2.cvtColor(im2, cv2.COLOR_GRAY2RGB), contours, -1, (255,255,0), 3)
				# cv2.imshow('cont', cont)
				# cv2.waitKey(0)
				yprob = readNumber(myRatio, minLoc, mainImage)

				information["digitConfidences"] = yprob.tolist()
				#print "yprob looks like ", yprob
				pictureNumber = yprob.argmax()/10
				guessedDigit = yprob.argmax() % 10
				maxConf = float(yprob.max())
				#print "max number is ", yprob.max(), " on picture ", pictureNumber
				#print "we predict it is ", guessedDigit
				information["digitPictureNumber"] = pictureNumber
				information["numberOfPictures"] = yprob.shape[0]
				information["maxDigitConf"] = maxConf
				information["predictedDigit"] = guessedDigit
					
					
				print information


				listOfDicts.append(information)

				# with open("notdoingatestnow.txt", "w") as myfile:
				# 	json.dump(listOfDicts, myfile)

			imageNumber += 1
				#cv2.imshow("step", color)
			#cv2.waitKey(0)
				# cv2.imshow("before", mainImage) 
				# justForShow = mainImage.copy()
				# justForShow[minLoc[0]:minLoc[0] + myHappyTemplate.shape[0], minLoc[1]:minLoc[1] + myHappyTemplate.shape[1]] += myHappyTemplate
				# cv2.imshow("replaced", justForShow) 
				# cv2.imwrite("justForShow.tiff", justForShow)

				
				#cv2.imshow("result", result)
				#cv2.waitKey(0)

				#mainImage[minMaxLoc[2][0] : minMaxLoc[2][0]+template.shape[0] , minMaxLoc[2][1] : minMaxLoc[2][1]+template.shape[1] ] = tempThresh
				#cv2.imshow("mainImage", mainImage)
				#cv2.waitKey(0)
		except (KeyboardInterrupt, SystemExit):
			raise
		except:
			print "there was some sort of error but we shall prevail! On image ", imageFileName


