import matplotlib.pyplot as plt
import cv2
import numpy as np


def region_of_interest(img,vertices):
    # Apply frame masking and find region of interest
    mask = np.zeros_like(img)
    mask_color = 255
    # create polygons
    cv2.fillPoly(mask,vertices,mask_color)
    masked_image = cv2.bitwise_and(img,mask)
    return masked_image


# draw lines on image
def draw_lines(image,lines):
    image = np.copy(image)
    # create new blank image with same size
    blank_image = np.zeros((image.shape[0],image.shape[1],3),dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            # on blank image draw lines and choose color 
            cv2.line(blank_image,(x1,y1),(x2,y2),(0,255,0),thickness=6) 
    # combine original image and drawn lines
    image = cv2.addWeighted(image,0.8,blank_image,1,0.0)
    return image



def process_image(image):
    # transform colors
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    # create vertices 
    imshape = image.shape
    top_left = [imshape[1]/2-imshape[1]/4,imshape[0]/2+imshape[0]/9]
    top_right = [imshape[1]/2+imshape[1]/4,imshape[0]/2+imshape[0]/9]
    lower_left = [0,imshape[0]]
    lower_right = [imshape[1],imshape[0]]
    vertices = np.array([[lower_left,top_left,top_right,lower_right]],dtype=np.int32)
    
    # turn image gray for easier detection 
    image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 

    # use canny edge detection
    image_canny = cv2.Canny(image_gray,100,200)

    # crop image according to vertices
    cropped_image = region_of_interest(image_canny,vertices)
    
    # apply Hough transform to image
    lines = cv2.HoughLinesP(cropped_image,rho=8,theta=np.pi/180,threshold=140,lines=np.array([]), minLineLength=40,maxLineGap=10)

    # apply lines on image
    img_with_lines=draw_lines(image,lines)
    return img_with_lines




cap = cv2.VideoCapture('driving-through-Red-Rock-State-Park.mp4')
# cap = cv2.VideoCapture('Dashcam.mp4')
while (cap.isOpened()):
    ret,frame =cap.read()
    frame = process_image(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)&0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # close all