import cv2
import requests
import numpy as np
def main():
    url = 'http://192.168.1.4:8080/shot.jpg'
    
    while True:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
        img  = cv2.imdecode(img_arr,-1)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #red range
        red_lower=np.array([136,87,111],np.uint8)
        red_upper=np.array([180,255,255],np.uint8)
        #blue range
        blue_lower=np.array([100,50,50],np.uint8)
        blue_upper=np.array([140,255,255],np.uint8)
        #yellow range
        yellow_lower=np.array([22,60,200],np.uint8)
        yellow_upper=np.array([60,255,255],np.uint8)
        #finding the range of red,blue and yellow color in the image
        red=cv2.inRange(hsv, red_lower, red_upper)
        blue=cv2.inRange(hsv,blue_lower,blue_upper)
        yellow=cv2.inRange(hsv,yellow_lower,yellow_upper)
        
        res=cv2.bitwise_and(img, img, mask = red)
        res1=cv2.bitwise_and(img, img, mask = blue)
        res2=cv2.bitwise_and(img, img, mask = yellow) 
        
        (_,contours,hierarchy)=cv2.findContours(red,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #Tracking the Red Color
        for pic, contour in enumerate(contours):
		        area = cv2.contourArea(contour)
		        if(area>300):
			
			           x,y,w,h = cv2.boundingRect(contour)	
			           img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
			           cv2.putText(img,"RED color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255))
                       
        (_,contours,hierarchy)=cv2.findContours(blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
			
	#Tracking the Blue Color
        for pic, contour in enumerate(contours):
		          area = cv2.contourArea(contour)
		          if(area>300):
			           x,y,w,h = cv2.boundingRect(contour)	
			           img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
			           cv2.putText(img,"Blue color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0))

	#Tracking the yellow Color
        (_,contours,hierarchy)=cv2.findContours(yellow,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        for pic, contour in enumerate(contours):
		         area = cv2.contourArea(contour)
		         if(area>300):
			           x,y,w,h = cv2.boundingRect(contour)	
			           img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
			           cv2.putText(img,"yellow  color",(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))  
        
        cv2.imshow("Color Tracking",img)
        #cv2.imshow("red",res)
        #cv2.imshow("blue",res1)
        #cv2.imshow("yellow",res2)
        if cv2.waitKey(1)==27:
            break
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()