import numpy as np
import cv2
import matplotlib.pyplot as plt
import keyboard

cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    a=1
    plt.imshow(frame)
    plt.show()

cap.release()
cv2.destroyAllWindows()
'''
a = np.array(images[0][0, :, 0, :, :])
b = np.array(images[1][0, :, 0, :, :])
c = np.array(images[2][0, :, 0, :, :])
a = np.moveaxis(a, 0, -1)
b = np.moveaxis(b, 0, -1)
c = np.moveaxis(c, 0, -1)
plt.imshow(a)
plt.show()
plt.imshow(b)
plt.show()
plt.imshow(c)
plt.show()


--dist-url tcp://localhost:10055 --multiprocessing-distributed --world-size 100 --rank 0 --fix-pred-lr --resume /home/kiran/kiran/Thesis/code/kiran_code/checkpoint_0006.pth.tar --start-epoch 7
'''