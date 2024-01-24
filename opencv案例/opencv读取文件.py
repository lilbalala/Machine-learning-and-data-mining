import cv2
img = cv2.imread('data/lena.jpg')
cv2.imshow("read_img",img)
cv2.waitKey(3000)
cv2.destroyAllWindows()