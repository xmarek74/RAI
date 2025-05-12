#imports
import sys
import os
from keras._tf_keras.keras import models ,layers ,optimizers, preprocessing
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit, QLabel, QVBoxLayout, QWidget, QFileDialog, QDesktopWidget, QMessageBox, QHBoxLayout, QDialog
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QStandardPaths
from os.path import dirname
import numpy as np

#class for pop-up after loading model
class CustomDialog(QDialog):
    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        #init atributes
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setFixedSize(400, 160)

        layout = QVBoxLayout()
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(10)

        #set up horizontal layout (flex look-a-like) for icon and text
        innerLayout = QHBoxLayout()
        iconLabel = QLabel()
        icon = QIcon("../public/icons/ok.png")
        iconPixmap = icon.pixmap(65, 65)
        iconLabel.setPixmap(iconPixmap)
        iconLabel.setFixedWidth(65)

        messageLabel = QLabel(message)
        messageLabel.setObjectName("messageText")
        messageLabel.setWordWrap(True)
        messageLabel.setAlignment(Qt.AlignVCenter | Qt.AlignRight)

        innerLayout.addWidget(iconLabel)
        innerLayout.addWidget(messageLabel, 1)
        

        layout.addLayout(innerLayout)

        #confirmation button
        okButton = QPushButton("Continue")
        okButton.setObjectName("messageButton")
        okButton.clicked.connect(self.accept)
        okButton.setCursor(Qt.PointingHandCursor)
        layout.addWidget(okButton, alignment=Qt.AlignRight)

        self.setLayout(layout)

#main class representing gui
class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        #init attributes
        self.model = None
        self.filePath = None
        self.result = None
        #set up path for loading images
        self.dir = QStandardPaths.writableLocation(QStandardPaths.DesktopLocation)
        #set up window centered
        self.setWindowTitle("Plant Disease Detection")
        self.setGeometry(0, 0, 400, 600)
        screen = QDesktopWidget().availableGeometry()
        screenWidth = screen.width()
        screenHeight = screen.height()
        self.move((screenWidth - 400) // 2, (screenHeight-600) // 2)
        #create layout of app
        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)
        layout = QVBoxLayout(centralWidget)
        layout.setContentsMargins(40, 20, 40, 20)
        layout.setSpacing(20)
        #add components to layout
        self.buttonLoad = QPushButton("Load image", self)
        self.buttonLoad.setObjectName("guiButton")
        self.buttonLoad.setCursor(Qt.PointingHandCursor)
        layout.addWidget(self.buttonLoad)

        self.buttonLoadModel = QPushButton("Load model", self)
        self.buttonLoadModel.setObjectName("guiButton")
        self.buttonLoadModel.setCursor(Qt.PointingHandCursor)

        layout.addWidget(self.buttonLoadModel)

        self.imageLabel = QLabel(self)
        self.imageLabel.setObjectName("imageArea")
        self.imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.imageLabel)

        self.buttonAnalyze = QPushButton("Analyze image", self)
        self.buttonAnalyze.setEnabled(False)
        self.buttonAnalyze.setObjectName("guiButton")
        self.buttonAnalyze.setCursor(Qt.PointingHandCursor)

        layout.addWidget(self.buttonAnalyze)

        #connect compononents to functions
        self.buttonLoad.clicked.connect(self.loadImage)
        self.buttonAnalyze.clicked.connect(self.analyzeImage)
        self.buttonLoadModel.clicked.connect(self.loadModel)

    def loadImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", self.dir, "Images Files (*.png *.jpg *.jpeg)")
        if fileName:
            pixmap = QPixmap(fileName)
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            if (self.model):
                self.buttonAnalyze.setEnabled(True)
            self.filePath = fileName
            #set last used as current location
            self.dir = dirname(fileName)
    def loadModel(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Model", self.dir, "Keras files (*.h5 *.keras)")
        if fileName:
            self.model = fileName
            alert = CustomDialog("Success", "Model has been loaded successfully", self)
            if (self.filePath):
                self.buttonAnalyze.setEnabled(True)
        else:
            #TODO test if error
            alert = CustomDialog("Error", "Model couldn't be loaded", self)
        alert.exec_()

    def analyzeImage(self):
        #static for now
        model = models.load_model(self.model)
        
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