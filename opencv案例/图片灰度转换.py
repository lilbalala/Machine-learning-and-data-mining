import cv2
img = cv2.imread('data/lena.jpg')
cv2.imshow("RGB",img)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray_img)
cv2.waitKey(0)
cv2.imwrite("gray_lena.jpg",gray_img)
cv2.destroyAllWindows()