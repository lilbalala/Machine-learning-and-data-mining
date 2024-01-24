import cv2
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read("trainer.yml")
# 准备识别的图片
img = cv2.imread("data/lena.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier(r"E:\Python\python3.12.0\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
faces = face_detector.detectMultiScale(gray)
for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # 人脸识别
    id, confidence = recogizer.predict(gray[y:y + h, x:x + w])
    print("标签id：", id, "置信度评分：", confidence)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()