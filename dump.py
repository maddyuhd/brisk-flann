import cv2
# from helper import imresize, imd

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# detector = cv2.BRISK_create(50,0,.5)
detector = cv2.BRISK_create(50,1,1.0)
star = cv2.xfeatures2d.StarDetector_create(15, 30, 10, 8, 5)

def imresize(image, w, h,val=320):
	ar=w/float(h)
	if h>w:
		ar=w/float(h)
		newH=val
		newW=int(newH*ar)
	elif h<w:
		ar=h/float(w)
		newW=val
		newH=int(newW*ar)
	else:
		newH=val
		newW=val
				
	img = cv2.resize(image, (newW, newH)) 
	return img;

def imd(img):
	(h,w) = img.shape[:2]
	return (h,w)

def show(kp, img):
	img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
	for k in kp:
		x, y = k.pt
		cv2.line(img, (int(x)-2,int(y)), (int(x)+2,int(y)),(0,252,248),1)
		cv2.line(img, (int(x),int(y)+2), (int(x),int(y)-2),(0,252,248),1)
	# cv2.imshow('ImageWindow',img)
	cv2.imwrite("imagename_Brief.jpg",img)

def brief_brisk(imagepath,val):

	src_img = cv2.imread(imagepath, 0)
	h,w = imd(src_img)
	src_img = imresize(src_img, w, h, val)

	kp = detector.detect(src_img, None)
	# kp = star.detect(src_img)
	kp, des = brief.compute(src_img, kp)
	kp_val = [] 
	show(kp,src_img)
	for point in kp: 
		temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
		kp_val.append(temp)

	return kp_val,des
	# return kp,des
