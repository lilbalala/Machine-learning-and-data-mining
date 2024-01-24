import cv2
# 加载图片
img = cv2.imread('data/lena.jpg')
# 图片转化为灰度图片
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 加载特征数据 CascadeClassifier级联分类器
face_detector = cv2.CascadeClassifier("E:\Python\python3.12.0\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
faces = face_detector.detectMultiScale(gray_image)
for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()