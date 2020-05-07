import argparse
from copy import deepcopy
from math import ceil
import cv2
import numpy as np
import matplotlib.pyplot as plt

def exposure_threshold(energy_curve):
    et = np.zeros(energy_curve.shape[1])
    for i in range(energy_curve.shape[1]):
        num = 0
        den = 0
        for j in range(energy_curve.shape[0]):
            num += j * energy_curve[j, i]
            den += energy_curve.shape[0] * energy_curve[j, i]
        exposure = (num / den)
        et[i] = (energy_curve.shape[0] - 1) * (1 - exposure)
    return et

def energy_curve_clipping(energy_curve):
    energy_curve = deepcopy(energy_curve)
    T_avg = energy_curve.mean(axis=0)
    for i in range(T_avg.shape[0]):
        temp = energy_curve[:, i]
        temp[temp > T_avg[i]] = T_avg[i]
    return energy_curve

def transfer_func(energy_curve, exp_thresh):
    transfer_funcn = np.zeros(energy_curve.shape)
    pdf = deepcopy(energy_curve)
    for i in range(pdf.shape[1]):
        pdf[:ceil(exp_thresh[i]), i] = np.divide(pdf[:ceil(exp_thresh[i]), i], \
            np.sum(pdf[:ceil(exp_thresh[i]), i])) * exp_thresh[i]
        pdf[ceil(exp_thresh[i]):, i] = (pdf.shape[0] - (exp_thresh[i] + 1)) * \
            np.divide(pdf[ceil(exp_thresh[i]):, i], np.sum(pdf[ceil(exp_thresh[i]):, i]))
        tf = 0
        for j in range(ceil(exp_thresh[i])):
            tf += pdf[j, i]
            transfer_funcn[j, i] = tf
        tf = ceil(exp_thresh[i])
        for j in range(ceil(exp_thresh[i]), pdf.shape[0]):
            tf += pdf[j, i]
            transfer_funcn[j, i] = tf
    return transfer_funcn

neighbourhood_matrix = np.array([[0, 1, 0],
                                 [1, 0, 1],
                                 [0, 1, 0]])

parser = argparse.ArgumentParser(description='Energy Curve Equalization')
parser.add_argument('input_file', metavar='I', type=str, nargs='+',
    help='path to input file')
parser.add_argument('-o', '--output_file', type=str, default='output.jpg')
parser.add_argument('--display_graphs', action='store_true', default=False)
args = parser.parse_args()

img = cv2.imread(args.input_file[0])
img -= np.min(img)
img = img.astype(np.float32)
img *= 255 / np.max(img)
img = img.astype(np.uint8)

if len(img.shape) == 2:
    channels = 1
    img = np.expand_dims(img, axis=-1)
else:
    channels = img.shape[2]

B_l = np.zeros([256, img.shape[0], img.shape[1], channels])
B_l -= 1

for i in range(256):
    temp = B_l[i]
    temp[img > i] = 1

E_l = np.zeros([256, channels])

# Do in GPU
for i in range(1, img.shape[0]-1):
    for j in range(1, img.shape[1]-1):
        E_l -= B_l[:, i, j, :] * np.sum(np.multiply(np.swapaxes(B_l[:, i-1:i+2, j-1:j+2, :], 1, 3),\
            np.swapaxes(neighbourhood_matrix, 0, 1)), axis=(2, 3))

E_l += (img.shape[0] - 2) * (img.shape[1] - 2) * np.sum(neighbourhood_matrix)

exp_thresh = exposure_threshold(E_l)

E_l_clip = energy_curve_clipping(E_l)

tf = transfer_func(E_l_clip, exp_thresh)

new_img = deepcopy(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for c in range(channels):
            new_img[i, j, c] = tf[img[i, j, c], c]

cv2.imwrite(args.output_file, new_img)

if args.display_graphs:
    plt.figure(1)
    if channels == 1:
        plt.plot(E_l[:, 0], '#999999')
        plt.xlabel('Pixel Value')
        plt.ylabel('Energy')
        plt.title('Input Image Energy Curve')
        
        plt.figure(2)
        plt.plot(E_l_clip[:, 0], '#999999')
        plt.xlabel('Pixel Value')
        plt.ylabel('Energy')
        plt.title('Clipped Energy Curve')
        
        plt.figure(3)
        plt.plot(tf[:, 0], '#999999')
        plt.xlabel('Input Pixel Value')
        plt.ylabel('Output Pixel Value')
        plt.title('Transfer Function')
    if channels == 3:
        plt.plot(E_l[:, 0], 'b')
        plt.plot(E_l[:, 1], 'g')
        plt.plot(E_l[:, 2], 'r')
        plt.xlabel('Pixel Value')
        plt.ylabel('Energy')
        plt.title('Input Image Energy Curve')

        plt.figure(2)
        plt.plot(E_l_clip[:, 0], 'b')
        plt.plot(E_l_clip[:, 1], 'g')
        plt.plot(E_l_clip[:, 2], 'r')
        plt.xlabel('Pixel Value')
        plt.ylabel('Energy')
        plt.title('Clipped Energy Curve')

        plt.figure(3)
        plt.plot(tf[:, 0], 'b')
        plt.plot(tf[:, 1], 'g')
        plt.plot(tf[:, 2], 'r')
        plt.xlabel('Input Pixel Value')
        plt.ylabel('Output Pixel Value')
        plt.title('Transfer Function')
    plt.show()
