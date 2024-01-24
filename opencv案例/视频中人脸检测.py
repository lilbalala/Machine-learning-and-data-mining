import cv2
video_face = cv2.VideoCapture("data/video.mp4")
while True:
    # read()方法返回视频中检测的对象，,视频在播放flag为True,frame为当前帧上的图片
    flag, frame = video_face.read()
    print("flag:", flag, "frame.shape:", frame.shape)
    if not flag:
        break
    # 将图片进行灰度化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 加载特征数据
    face_detector = cv2.CascadeClassifier("E:\Python\python3.12.0\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        cv2.circle(frame, center=(x + w // 2, y + h // 2), radius=(w // 2), color=(0, 255, 0), thickness=2)
    cv2.imshow("result", frame)
    cv2.waitKey(20)
cv2.destroyAllWindows()
video_face.release()