import copy
import os
import cv2

path_dir: str ="D:/hastaCropping"
list_photo = (os.listdir(path_dir))

for i in list_photo:
    IMG_IN = "D:/hastaCropping/"+i

    original = cv2.imread(IMG_IN)

# Read the image, convert it into grayscale, and make in binary image for threshold value
    img = cv2.imread(IMG_IN,0)

# use binary threshold, all pixel that are beyond 3 are made white
    _, thresh_original = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)

# Now find contours in it.
    thresh = copy.copy(thresh_original)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# get contours with highest height
    lst_contours = []
    for cnt in contours:
        ctr = cv2.boundingRect(cnt)
        lst_contours.append(ctr)
    x,y,w,h = sorted(lst_contours, key=lambda coef: coef[3])[-1]

    ctr = copy.copy(original)
    cv2.rectangle(ctr, (x,y),(x+w,y+h),(0,255,0),2)

    img2 = cv2.imread(IMG_IN)
    crop_img = img2[y:y + h, x:x + w]
    cv2.imwrite(IMG_IN, crop_img)



