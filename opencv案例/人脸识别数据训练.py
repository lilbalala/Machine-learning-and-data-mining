def out_getImageAndLabels():
    import cv2
    import numpy as np
    def getImageAndLabels(path):
        # 导包
        import os
        import sys
        from PIL import Image
        facesSamples = []
        ids = []
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        face_detector = cv2.CascadeClassifier(r"E:\Python\python3.12.0\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")

        # 遍历列表中的图片
        for imagePath in imagePaths:
            PIL_img = Image.open(imagePath).convert("L")
            # 将图像转换我数组
            img_numpy = np.array(PIL_img, "uint8")
            faces = face_detector.detectMultiScale(img_numpy)
            # 获取每张图片的id
            id = int(os.path.split(imagePath)[1].split(".")[0])
            # id = os.path.split(imagePath)[1].split(".")[0]
            for x, y, w, h in faces:
                # 添加人脸区域图片
                facesSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)
        return facesSamples, ids

    # 图片路径
    path = "data/jm"
    # 获取图像数组和id数组标签
    faces, ids = getImageAndLabels(path)
    # 训练对象
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    # 保存训练文件
    recognizer.write("trainer.yml")