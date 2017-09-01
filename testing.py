from subprocess import call 

call(["tesseract", "../myNewFaxes/590a1bf03839be4d17b4bebe", "../myTestOutput", "-l", "eng", "--psm", "0"])

with open('../myTestOutput.osd') as myFile:
	output = myFile.read()

print output