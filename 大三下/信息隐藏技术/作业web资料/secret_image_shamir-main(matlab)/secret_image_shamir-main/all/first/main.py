#International Journal of Applied Engineering Research ISSN 0973-4562 Volume 13, Number 4 (2018) pp. 1827-1832
# © Research India Publications. http://www.ripublication.com
# 1827
# Design and Performance Evaluation of Boolean based Secret Image Sharing Scheme
# Javvaji V.K. Ratnam1, T. Sreenivasulu Reddy2 and P. Ramana Reddy3
# 1Research Scholar, Faculty of Electronics and Communication Engineering,
# Jawaharlal Nehru Technological University Anantapur, Ananthapuramu, Andhra Pradesh, India.
# 1Orcid: 0000-0003-3099-6673
# 2Professor, Department of Electronics and Communication Engineering,
# Sri Venkateswara University College of Engineering, Sri Venkateswara University, Tirupati, Andhra Pradesh, India.
# 3Professor, Department of Electronics and Communication Engineering, University College of Engineering,
# Jawaharlal Nehru Technological University Anantapur, Ananthapuramu, Andhra Pradesh, India.
import cv2
image = cv2.imread("face.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

cv2.imshow("Original image",gray)
cv2.waitKey()

import numpy as np
import random

X = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.uint8)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i][j] = random.randint(1,255)


Y = np.bitwise_xor(gray,X)

cv2.imshow("original image xor",X)
cv2.waitKey()

#B = np.zeros((gray.shape[0],gray.shape[1]),dtype=np.uint8)
B  = Y//2


n = 3

i= 0
ran = []
aunth_id = []

while i<n:
    ran.append(random.randint(0,255))
    b = (bin(ran[i]).replace("0b", ""))
    x1 = (b[len(b)-4:len(b)])
    x1 = int(x1, 2)
    x2 = b[0:4]
    x2 = int(x2,2)
    aunth_id.append(x1^x2)
    i+=1

print(aunth_id)
S1 = np.roll(B, aunth_id[0], axis=1)
S2 = np.roll(B, aunth_id[1], axis=1)
S3 = np.roll(B, aunth_id[2], axis=1)     




cv2.imshow("intermediate step ",S1)
cv2.waitKey()



P = np.roll(S1, -aunth_id[0], axis=1)
P+= np.roll(S2, -aunth_id[1], axis=1)

print(P.shape)


G1 = np.bitwise_xor(P,X)

cv2.imshow("Final output ",G1)
cv2.waitKey()














