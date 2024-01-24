import cv2
# 加载图片
img = cv2.imread("data/face3.jpg")
# 图片进行灰度处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 加载数据特征
face_detector = cv2.CascadeClassifier("E:\Python\python3.12.0\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
faces = face_detector.detectMultiScale(gray)
for x, y, w, h in faces:
    print(x, y, w, h)
    cv2.rectangle(img, (x, y), (x + w, y + h),
                  color=(0, 0, 255), thickness=2)
    cv2.circle(img, (x + w // 2, y + w // 2),
               radius=w // 2, color=(0, 255, 0), thickness=2)
cv2.imshow("result", img)
cv2.waitKey(0)