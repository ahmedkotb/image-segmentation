#!/usr/bin/env python
import sip
sip.setapi('QVariant', 2)

from math import cos, pi, sin

from PyQt4 import QtCore, QtGui


class RenderArea(QtGui.QWidget):
    def __init__(self, path, parent=None):
        super(RenderArea, self).__init__(parent)

        self.path = path

        self.penWidth = 1
        self.rotationAngle = 0
        self.setBackgroundRole(QtGui.QPalette.Base)

    def minimumSizeHint(self):
        return QtCore.QSize(50, 50)

    def sizeHint(self):
        return QtCore.QSize(100, 100)

    def setFillRule(self, rule):
        self.path.setFillRule(rule)
        self.update()

    def setFillGradient(self, color1, color2):
        self.fillColor1 = color1
        self.fillColor2 = color2
        self.update()

    def setPenWidth(self, width):
        self.penWidth = width
        self.update()

    def setPenColor(self, color):
        self.penColor = color
        self.update()

    def setRotationAngle(self, degrees):
        self.rotationAngle = degrees
        self.update()

class Window(QtGui.QWidget):
    NumRenderAreas = 9

    def __init__(self):
        super(Window, self).__init__()

        browseButton = self.createButton("&Browse...", self.browse)
        directoryLabel = QtGui.QLabel("Image:")
        self.algorithmComboBox = QtGui.QComboBox()
        self.algorithmComboBox.addItem("K-means")
        self.algorithmComboBox.addItem("Mean Shift")
        self.algorithmComboBox.addItem("Graph Cut")

        algorithm = QtGui.QLabel("&Algorithm:")
        algorithm.setBuddy(self.algorithmComboBox)

        self.featureComboBox = QtGui.QComboBox()
        self.featureComboBox.addItem("Intensity")
        self.featureComboBox.addItem("Intensity and pixel coordinates")
        self.featureComboBox.addItem("RGB color")
        self.featureComboBox.addItem("YUV color")
        self.featureComboBox.addItem("Invariant LM")
        self.featureComboBox.addItem("PCA")
        self.featureComboBox.addItem("Leung-Malik")
        feature = QtGui.QLabel("&Feature:")
        feature.setBuddy(self.featureComboBox)
        self.userpath = QtGui.QLineEdit()
        self.kText = QtGui.QLineEdit()

        self.kLabel = QtGui.QLabel("&K:")
        self.kLabel.setBuddy(self.kText)

        self.iterationsText = QtGui.QLineEdit()
        self.epsilonText = QtGui.QLineEdit()

        self.iterationsLabel = QtGui.QLabel("&Number Of Iterations:")
        self.iterationsLabel.setBuddy(self.iterationsText)

        self.epsilonLabel = QtGui.QLabel("&Epsilon:")
        self.epsilonLabel.setBuddy(self.epsilonText)

        self.featureComboBox.activated.connect(self.featureChanged)
        self.algorithmComboBox.activated.connect(self.algorithmChanged)


        topLayout = QtGui.QGridLayout()

        self.mainLayout = QtGui.QGridLayout()
        self.mainLayout.addLayout(topLayout, 0, 0, 1, 4)
        self.mainLayout.addWidget(algorithm, 1, 0)
        self.mainLayout.addWidget(self.algorithmComboBox, 1, 1, 1, 3)
        self.mainLayout.addWidget(feature, 2, 0)
        self.mainLayout.addWidget(self.featureComboBox, 2, 1)
        self.mainLayout.addWidget(self.kLabel, 4, 0)
        self.mainLayout.addWidget(self.kText, 4, 1, 1, 3)
        self.mainLayout.addWidget(self.iterationsLabel, 5, 0)
        self.mainLayout.addWidget(self.iterationsText, 5, 1, 1, 3)
        self.mainLayout.addWidget(browseButton, 3, 2)
        self.mainLayout.addWidget(directoryLabel, 3, 0)
        self.mainLayout.addWidget(self.userpath, 3, 1)
        self.mainLayout.addWidget(self.epsilonText, 6, 1)
        self.mainLayout.addWidget(self.epsilonLabel, 6, 0)
        self.setLayout(self.mainLayout)

        self.featureChanged()
        self.algorithmChanged()

        self.setWindowTitle("Segmentation")


    def browse(self):
        filePath = QtGui.QFileDialog.getOpenFileName(self, "Find Files",
                QtCore.QDir.currentPath())
        self.userpath.setText(filePath)

    def createButton(self, text, member):
        button = QtGui.QPushButton(text)
        button.clicked.connect(member)
        return button

    def createComboBox(self, text=""):
        comboBox = QtGui.QComboBox()
        comboBox.setEditable(True)
        comboBox.addItem(text)
        comboBox.setSizePolicy(QtGui.QSizePolicy.Expanding,
                QtGui.QSizePolicy.Preferred)
        return comboBox

    def featureChanged(self):
        feature = self.featureComboBox.currentText()

    def algorithmChanged(self):
        algorithm = self.algorithmComboBox.currentText()
        if algorithm == "Mean Shift":
          self.featureComboBox.removeItem(6)
          self.kText.hide()
          self.kLabel.hide()
          self.iterationsText.hide()
          self.iterationsLabel.hide()
          self.epsilonLabel.hide()
          self.epsilonText.hide()
        elif algorithm == "Graph Cut":
          self.featureComboBox.removeItem(6)
          self.kText.hide()
          self.kLabel.hide()
          self.iterationsText.hide()
          self.iterationsLabel.hide()
          self.epsilonLabel.hide()
          self.epsilonText.hide()
        elif algorithm == "K-means":
          if self.featureComboBox.count() == 6:
          	self.featureComboBox.addItem("Leung-Malik")
          self.kText.show()
          self.kLabel.show()
          self.iterationsText.show()
          self.iterationsLabel.show()
          self.epsilonLabel.show()
          self.epsilonText.show()
    def populateWithColors(self, comboBox):
        colorNames = QtGui.QColor.colorNames()
        for name in colorNames:
            comboBox.addItem(name, name)

    def currentItemData(self, comboBox):
        return comboBox.itemData(comboBox.currentIndex())


if __name__ == '__main__':

    import sys

    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
