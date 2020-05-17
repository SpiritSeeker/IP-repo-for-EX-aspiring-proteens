import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

#Reading the input

img = cv2.imread('lena_gray_dark.jpg')
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y, u, v = cv2.split(img_yuv)

y = y.astype(np.uint64)

#Acquiring Histogram

h = np.zeros((256,256))
print(img.shape)
for j in range(y.shape[0]):
    for i in range(y.shape[1]):
        a = y[j][i]
        if j != img.shape[0]-1:
            b = y[j+1][i]
            h[max(b,a)][min(b,a)] = h[max(b,a)][min(b,a)] + 1
        
        if i != img.shape[1]-1:
            b = y[j][i+1]
            h[max(b,a)][min(b,a)] = h[max(b,a)][min(b,a)] + 1

#Intra Layer optimisation            

D = np.zeros((255,255))
s = np.zeros(255)
pl = np.zeros((256,256)) #Used for plotting histogram

U = np.zeros([255,255])
for l in range(255):
    for k in range(255):
        U[l][k] = min(k+1,256-l-1) - max(k-l,0)
        

for layer in range(255):
    h_l = np.zeros(255-layer)
    temp = 0
    for j in range(layer,255,1):
        i = j-layer
        
        h_l[temp] = np.log(h[j][i]+1)
        pl[temp][layer] = h_l[temp]
        temp += 1
        
    s[layer] = np.sum(h_l)
    if s[layer]==0:
        continue
    one = np.ones(layer+1)
    ones = np.ones(255)
    m_l = np.convolve(h_l,one)
    d_l = np.dot(np.linalg.inv(np.diagflat(U[l])),(m_l-(min(m_l)*ones)))
    if np.sum(d_l)==0:
        continue
    D[layer] = d_l/np.sum(d_l)

#Inter Layer aggregation
    
alpha = 5.0
W = np.power((s/max(s)),alpha)
d = np.dot(D,W)
d = d/(np.sum(d))   

tmp = np.zeros(256)
print(tmp.shape)
for i in range(tmp.shape[0]-1):
    tmp[i+1] = tmp[i] + d[i]

x = 255*tmp
 
#Writing the output

out = np.zeros((img.shape[0],img.shape[1]))
for j in range(y.shape[0]):
    for i in range(y.shape[1]):
        out[j][i] = round(x[y[j][i]])
        
out = out.astype(np.uint8)
out_yuv = cv2.merge((out,u,v))

out_image = cv2.cvtColor(out_yuv,cv2.COLOR_YUV2BGR)



cv2.imwrite('lenaLDRout5.0.jpg', out_image)