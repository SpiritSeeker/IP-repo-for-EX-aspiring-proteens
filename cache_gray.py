from copy import deepcopy
import cv2
import numpy as np

def zero_interpolate(img):
	out = np.zeros([img.shape[0] * 2, img.shape[1] * 2])
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			out[2 * i, 2 * j] = img[i, j]
	return out

L = 4
S = 512
eps = 1e-3

neighbourhood_matrix = np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])

img = cv2.imread('lena_gray_dark.jpg', 0)
aspect_ratio = float(img.shape[1]) / float(img.shape[0])

if aspect_ratio < 1:
	width = S
	height = S / aspect_ratio

else:
	height = S
	width = S * aspect_ratio

a = []

a_i = cv2.resize(img, (int(height), int(width)), interpolation=cv2.INTER_CUBIC)

for i in range(L):
	a.append(a_i)
	a_i = cv2.resize(a_i, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

phi_tot = np.ones(a[0].shape)

for l in range(L):
	temp_img = cv2.copyMakeBorder(a[l], 1, 1, 1, 1, cv2.BORDER_REFLECT)
	temp_phi = np.zeros(a[l].shape)

	for i in range(temp_phi.shape[0]):
		for j in range(temp_phi.shape[1]):
			neighbourhood = (temp_img[i:i+3, j:j+3] - temp_img[i+1, j+1]) / 255
			neighbourhood[neighbourhood < 0] = 0
			neighbourhood = np.multiply(neighbourhood, neighbourhood_matrix)
			temp_phi[i, j] = np.sum(neighbourhood)

	for k in range(l):
		temp_phi = zero_interpolate(temp_phi)
	temp_phi[temp_phi == 0] = eps
	phi_tot = np.multiply(phi_tot, temp_phi)

phi_tot = phi_tot ** (1 / L)

phi_tot_sum = np.sum(phi_tot)

pdf = np.zeros(256)
cdf = np.zeros(256)
acc = 0

for i in range(256):
	kroneck = np.zeros(phi_tot.shape)
	kroneck[a[0] == i] = 1
	pdf_num = np.multiply(phi_tot, kroneck)
	pdf[i] = np.sum(pdf_num) / phi_tot_sum
	acc += pdf[i]
	cdf[i] = acc

cdf *= 255

new_img = np.zeros(a[0].shape)

for i in range(a[0].shape[0]):
	for j in range(a[0].shape[1]):
		new_img[i, j] = cdf[int(a[0][i, j])]

cv2.imwrite('saved.jpg', new_img)
