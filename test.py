import cv2
import numpy as np

black = np.zeros((360, 640, 3))

cv2.imshow("debug", black)
cv2.waitKey(0)
cv2.destroyAllWindows()
