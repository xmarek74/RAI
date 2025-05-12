#imports
import sys
from keras._tf_keras.keras import models ,layers ,optimizers, preprocessing
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget, QFileDialog, QDesktopWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QStandardPaths
from os.path import dirname
import numpy as np

#main class representing gui
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #set up path for loading images
        self.dir = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
        #set up window centered
        self.setWindowTitle("PyQt5 Plant Disease Detection")
        self.setGeometry(0, 0, 400, 400)
        screen = QDesktopWidget().availableGeometry()
        screenWidth = screen.width()
        screenHeight = screen.height()
        self.move((screenWidth - 300) // 2, (screenHeight-200) // 2)
        #create layout of app
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(20)
        #add components to layout
        self.buttonLoad = QPushButton("Load image", self)
        self.buttonLoad.setObjectName("loadButton")
        layout.addWidget(self.buttonLoad)

        self.imageLabel = QLabel(self)
        self.imageLabel.setObjectName("imageArea")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.imageLabel)

        self.buttonAnalyze = QPushButton("Analyze image", self)
        self.buttonAnalyze.setEnabled(False)
        layout.addWidget(self.buttonAnalyze)

        #connect compononents to functions
        self.buttonLoad.clicked.connect(self.loadImage)
        self.buttonAnalyze.clicked.connect(self.analyzeImage)

    def loadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", self.dir, "Images Files (*.png *.jpg *.bmp *.jpeg)")
        if fileName:
            pixmap = QPixmap(fileName)
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            self.buttonAnalyze.setEnabled(True)
            self.filePath = fileName
            #set last used as current location
            self.dir = dirname(fileName)
    #TODO connect with main CNN
    def analyzeImage(self):
        #static for now
        model = models.load_model("diseaseModel.h5")
        file = preprocessing.image.load_img(self.filePath, target_size=(128, 128))
        #normalize rgb
        fileArray = preprocessing.image.img_to_array(file) / 255.0
        #add dim
        fileArray = np.expand_dims(fileArray, axis=0)
        prediction = np.argmax(model.predict(fileArray), axis=1)
        #0 means disased 1 means healthy
        self.result = "healthy" if (prediction[0]) else "diseased"
        print(f"{self.result}")


def main():
    app = QApplication(sys.argv)
    window = MyWindow()
    
    with open("stylesheet.css", "r") as file:
        style = file.read()
        app.setStyleSheet(style)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()