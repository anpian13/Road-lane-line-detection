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



def process_image(image,dash):
    # transform colors
    
    # create vertices 
    imshape = image.shape
    
    # if case of dashcam region of interest changes:
    # broaden top horizontal vertices points so all lanes visible
    # set lower points higher so dashboard wouldn't be in region

    
    
    # dash = False

    if dash:
        offset_top_horizontal = imshape[1]/4
        offset_bottom_vert = imshape[0]/4

    else:
        offset_top_horizontal = 0
        offset_bottom_vert = 0



    top_left = [imshape[1]/2-imshape[1]/4 - offset_top_horizontal,imshape[0]/2+imshape[0]/9]
    top_right = [imshape[1]/2+imshape[1]/4 + offset_top_horizontal,imshape[0]/2+imshape[0]/9]
    lower_left = [0,imshape[0] - offset_bottom_vert]
    lower_right = [imshape[1],imshape[0] - offset_bottom_vert]
    vertices = np.array([[lower_left,top_left,top_right,lower_right]],dtype=np.int32)
    
    # turn image gray for easier detection 
    image_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) 

    # use canny edge detection

    image_canny = cv2.Canny(image_gray,150,200)

    # crop image according to vertices
    cropped_image = region_of_interest(image_canny,vertices)
    
    # apply Hough transform to image
    lines = cv2.HoughLinesP(cropped_image,rho=6,theta=np.pi/180,threshold=140,lines=np.array([]), minLineLength=40,maxLineGap=10)

    # apply lines on image
    img_with_lines=draw_lines(image,lines)
    return img_with_lines



video_name = 'driving-through-Red-Rock-State-Park.mp4'
# video_name = 'RAW-DASH-CAM.mp4'

# checking if video from dashcam based on file name
if 'dash' in video_name.lower():
    dash_flag = True
else: dash_flag = False

cap = cv2.VideoCapture(video_name)
# cap = cv2.VideoCapture('Dashcam.mp4')
while (cap.isOpened()):
    ret,frame =cap.read()
    frame = process_image(frame,dash_flag)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)&0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows() # close all