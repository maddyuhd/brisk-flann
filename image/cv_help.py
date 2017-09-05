import cv2

# brief = cv2.xfeatures2d.BriefDescriptorExtractor_create() <
# detector = cv2.BRISK_create(50,0,.5)

# detector = cv2.BRISK_create(50, 1, 1.0)
# star = cv2.xfeatures2d.StarDetector_create(15, 30, 10, 8, 5)


def imresize(image, w, h, val=320):
    ar = w / float(h)
    if h > w:
        ar = w / float(h)
        newH = val
        newW = int(newH * ar)
    elif h < w:
        ar = h / float(w)
        newW = val
        newH = int(newW * ar)
    else:
        newH = val
        newW = val

    img = cv2.resize(image, (newW, newH))
    return img


def show(kp, img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for k in kp:
        x, y = k.pt
        cv2.line(img, (int(x) - 2, int(y)), (int(x) + 2, int(y)),
                 (0, 252, 248), 1)
        cv2.line(img, (int(x), int(y) + 2), (int(x), int(y) - 2),
                 (0, 252, 248), 1)
    # cv2.imshow('ImageWindow', img)
    cv2.imwrite("imagename_Brief.jpg", img)


def briefBrisk(imagepath, val, src=True, debug=False):

    src_img = cv2.imread(imagepath, 0)
    h, w = src_img.shape[:2]
    src_img = imresize(src_img, w, h, val)

    if src:
        detector = cv2.ORB_create(400, 1.2, 8, 5, 0, 2, 0, 31, 20)
        # detector = cv2.ORB_create()
        # detector = cv2.BRISK_create(50, 1, 1.0) <

    else:
        detector = cv2.ORB_create(100, 2, 9, 50, 0, 2, 0, 31, 5)
        # detector = cv2.xfeatures2d.StarDetector_create(10, 30, 15) <
        # detector = cv2.xfeatures2d.StarDetector_create(15, 30, 10, 8, 5)
        # detector = cv2.BRISK_create(110, 1, 1.0)

    kp = detector.detect(src_img, None)

    kp, des = detector.compute(src_img, kp)
    # kp, des = brief.compute(src_img, kp) <
    kp_val = []

    if not src:
        show(kp, src_img)

    for point in kp:
        temp = (point.pt, point.size, point.angle,
                point.response, point.octave, point.class_id)
        kp_val.append(temp)

    return kp_val, des
    # return kp, des
