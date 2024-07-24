import numpy as np 
import cv2 
from math import log10, sqrt 

image = cv2.imread("face.jpg")

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey()

R1 = gray//3+15
R2 = gray//3+1
R3 =gray//3+0
R4 = np.random.randint(255,size = (gray.shape[0],gray.shape[1]),dtype = np.uint8)

S1 = np.bitwise_xor(R1,R4)
S2 = np.bitwise_xor(R2,R4)
S3 = np.bitwise_xor(R3,R4)

S4  = R4

cv2.imshow("secret image 1",S1)
cv2.imwrite("encrypted_image1.jpg",S1)
cv2.waitKey()
cv2.imshow("secret image 2",S2)
cv2.imwrite("encrypted_image2.jpg",S2)
cv2.waitKey()
cv2.imshow("secret image 3",S3)
cv2.imwrite("encrypted_image3.jpg",S3)
cv2.waitKey()
cv2.imshow("secret image 4",S4)
cv2.imwrite("encrypted_image4.jpg",S4)
cv2.waitKey()


#k==n
def PSNR(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr

G = np.bitwise_xor(S1,S4)+np.bitwise_xor(S2,S4)+np.bitwise_xor(S3,S4)
G-=16

if gray.all()==G.all():
    print(True)

cv2.imshow("final",G)
cv2.waitKey()

#k<n , k==2

G1 = np.bitwise_xor(S1,S2)+np.bitwise_xor(S2,S3)+np.bitwise_xor(S1,S3)
G1%=256
print(PSNR(gray, G1))
cv2.imshow("final_noise_image",G1)
cv2.imwrite("final_noise_image.jpg",G1)
cv2.waitKey()
converted_img = cv2.cvtColor(G1, cv2.COLOR_GRAY2BGR)
dst = cv2.fastNlMeansDenoisingColored(converted_img, None, 10, 10, 7, 15)
g = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
print(PSNR(gray, g))
cv2.imshow("final_denoise_image",g)
cv2.imwrite("final_after_denoise_image.jpg",g)
cv2.waitKey()


































