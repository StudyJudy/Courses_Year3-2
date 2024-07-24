import numpy as np 
import cv2 


image = cv2.imread("face.jpg")

T = np.random.randint(255,size = (image.shape[0],image.shape[1]),dtype = np.uint8)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#B = np.random.randint(255,size = (image.shape[0],image.shape[1]))

B1 = np.bitwise_xor(gray,T)
cv2.imshow("secret image",B1)
cv2.imwrite("encrypted_image2.jpg",B1)
cv2.waitKey()

# generating secret image 
B2 = np.bitwise_xor(B1,T)

cv2.imshow("final output",B2)
cv2.imwrite("final_output.jpg",B2)
cv2.waitKey()

