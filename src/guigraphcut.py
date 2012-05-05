#!/usr/bin/env python

import sys
import graphcut

from PyQt4 import QtCore, QtGui

IMG_PATH = "../lm_images/grayscale2.jpg"
IMG_PATH = "../lm_images/grayscale1_2.png"
IMG_PATH = "../test images/single object/189080.jpg"
IMG_PATH = "../test images/single object/285079.jpg"

RECT_WIDTH = 10
RECT_HEIGHT= 10
RECT_TRANS = 110

RECT_BACKGROUND_COLOR = (0,0,255)
RECT_OBJECT_COLOR = (255,0,0)

class Box(QtGui.QGraphicsRectItem):
    def __init__(self,x,y,w,h,parent=None,scene=None):
        QtGui.QWidget.__init__(self, x,y,w,h,parent,scene)
        self.color = QtGui.QColor(255,255,255,RECT_TRANS)

    def paint(self,painter,option,widget):
        x = self.rect().x()
        y = self.rect().y()
        w = self.rect().width()
        h = self.rect().height()
        painter.setPen(QtGui.QPen(QtGui.QColor(0,0,0,0)))
        painter.setBrush(self.color)
        painter.drawRect(x,y,w,h)

class BackgroundBox(Box):
    def __init__(self,x,y,w,h,parent=None,scene=None):
        QtGui.QWidget.__init__(self, x,y,w,h,parent,scene)
        c = RECT_BACKGROUND_COLOR
        self.color = QtGui.QColor(c[0],c[1],c[2],RECT_TRANS)

class ObjectBox(Box):
    def __init__(self,x,y,w,h,parent=None,scene=None):
        QtGui.QWidget.__init__(self, x,y,w,h,parent,scene)
        c = RECT_OBJECT_COLOR
        self.color = QtGui.QColor(c[0],c[1],c[2],RECT_TRANS)

class MainWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.boxes = []
        self.mode = "background"

        self.scene = QtGui.QGraphicsScene()
        self.view = QtGui.QGraphicsView(self.scene)
        layout = QtGui.QVBoxLayout()

        self.featureComboBox = QtGui.QComboBox()
        self.featureComboBox.addItem("Intensity")
        self.featureComboBox.addItem("Intensity and pixel coordinates")
        self.featureComboBox.addItem("RGB color")
        self.featureComboBox.addItem("YUV color")
        self.featureComboBox.addItem("Invariant LM")
        self.featureComboBox.addItem("PCA")
        layout.addWidget(self.featureComboBox)

        self.backbtn = QtGui.QPushButton("Background",self)
        self.objbtn = QtGui.QPushButton("Object",self)

        self.backbtn.setCheckable(True)
        self.backbtn.setChecked(True)
        self.objbtn.setCheckable(True)

        self.backbtn.clicked[bool].connect(self.changeMode)
        self.objbtn.clicked[bool].connect(self.changeMode)

        layout.addWidget(self.backbtn)
        layout.addWidget(self.objbtn)

        layout.addWidget(self.view)
        self.setLayout(layout)
        self.pixmap_item = QtGui.QGraphicsPixmapItem(QtGui.QPixmap(IMG_PATH)
                , None, self.scene)
        self.pixmap_item.mousePressEvent = self.pixelSelect

        self.segmentbtn = QtGui.QPushButton("Segment",self)
        layout.addWidget(self.segmentbtn)
        self.segmentbtn.clicked.connect(self.startSegmentation)

    def startSegmentation(self):
        imgW = self.pixmap_item.boundingRect().width()
        imgH = self.pixmap_item.boundingRect().height()
        print imgW,imgH

        background = []
        object = []

        for box in self.boxes:
            rect = box.boundingRect()
            x = int(rect.x())
            y = int(rect.y())
            w = int(rect.width())
            h = int(rect.height())

            for i in xrange(x,x + w):
                for j in xrange(y,y + h):
                    if j >= imgH or i >= imgW or i < 0 or j < 0:
                        continue
                    if type(box) == BackgroundBox:
                        background.append((j,i))
                    else:
                        object.append((j,i))

        #sort the pixels by the first coordinate then second cordinate
        #first coordinate  (0 --> image height)
        #second coordinate (0 --> image width)
        object.sort()
        background.sort()
        print "object pixels count",len(object)
        print "background pixels count",len(background)
        features = ["INTENSITY","INTENSITY+LOC","RGB","YUV","ILM","PCA"]
        selectedFeature = features[self.featureComboBox.currentIndex()]
        print selectedFeature
        graphcut.segmentUsingGraphcut(IMG_PATH, selectedFeature, object, background)

    def changeMode(self,event):
        source = self.sender()
        if not source.isChecked():
            return
        if source == self.backbtn:
            self.objbtn.setChecked(False)
            self.mode = "background"
        elif source == self.objbtn:
            self.backbtn.setChecked(False)
            self.mode = "object"

    def pixelSelect(self, event):
        found = False
        for box in self.boxes:
            if box.contains(event.pos()):
                found = True
                break
        if found == False:
            if self.mode == "background":
                self.boxes.append(BackgroundBox(event.pos().x() - RECT_WIDTH/2,
                    event.pos().y() - RECT_HEIGHT/2, RECT_WIDTH,RECT_HEIGHT,None,self.scene))
            elif self.mode == "object":
                self.boxes.append(ObjectBox(event.pos().x() - RECT_WIDTH/2,
                    event.pos().y() - RECT_HEIGHT/2, RECT_WIDTH,RECT_HEIGHT,None,self.scene))

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    widget = MainWidget()
    widget.resize(640, 480)
    widget.show()
    sys.exit(app.exec_())
