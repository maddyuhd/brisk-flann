import cv2
# from helper import imresize, imd

brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
detector = cv2.BRISK_create(50,0,.5)

imagepath = "image.jpg"
src_img = cv2.imread(imagepath)
h,w = src_img.shape[:2]
src_img = imresize(src_img,w,h,320)

kp = detector.detect(src_img, None)
kp, des = brief.compute(src_img, kp)
kp_val = [] 
for point in kp: 
	temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
	kp_val.append(temp)

return kp_val,des
