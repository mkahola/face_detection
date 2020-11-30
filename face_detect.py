#!/usr/bin/env python3

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtCore, QtGui

import cv2
import os
import sys

camera = 2 #dev/video0...2

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: black;")
        self.setWindowTitle("Face detection")
        self.video_size = QSize(864, 480)
        self.fps = 20.0
        self.resolution = QDesktopWidget().availableGeometry(-1)
        self.setup_ui()
        self.setup_camera()
        self.showFullScreen()

    def setup_ui(self):
        #Initialize widgets
        self.centralWidget = QWidget()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)

        centralLayout = QVBoxLayout()
        centralLayout.addWidget(self.image_label, 1)

        self.centralWidget.setLayout(centralLayout)
        self.setCentralWidget(self.centralWidget)

    def setup_camera(self):
        #Initialize camera
        self.timer = QTimer()
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.display_video_stream)

        self.capture = cv2.VideoCapture(camera)

        if not self.capture.isOpened():
            raise Exception("Camera unavailable")
            self.close()

        self.timer.start(1)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.video_size.width())
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.video_size.height())
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        self.cascPath = "haarcascade_frontalface_default.xml"

    def display_video_stream(self):
        #Read frame from camera and repaint QLabel widget
        # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(self.cascPath)
        check, image = self.capture.read()
        image = cv2.flip(image, 1)

        if check:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Detect faces in the image
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            image = QImage(image, image.shape[1], image.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image).scaled(self.resolution.width()-60, self.resolution.height()-60, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.capture.release()
            if self.timer.isActive():
                self.timer.stop()
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    sys.exit(app.exec_())
