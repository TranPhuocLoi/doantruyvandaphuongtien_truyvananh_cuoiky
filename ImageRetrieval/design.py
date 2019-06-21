# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'design.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from IR import *


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(937, 603)
        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setGeometry(QtCore.QRect(536, 285, 151, 121))
        self.label_4.setObjectName("label_4")
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setGeometry(QtCore.QRect(56, 415, 151, 121))
        self.label_11.setObjectName("label_11")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(216, 285, 151, 121))
        self.label_3.setObjectName("label_3")
        self.label_9 = QtWidgets.QLabel(Form)
        self.label_9.setGeometry(QtCore.QRect(376, 415, 151, 121))
        self.label_9.setObjectName("label_9")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(56, 285, 151, 121))
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(Form)
        self.label_5.setGeometry(QtCore.QRect(376, 285, 151, 121))
        self.label_5.setObjectName("label_5")
        self.label_10 = QtWidgets.QLabel(Form)
        self.label_10.setGeometry(QtCore.QRect(536, 415, 151, 121))
        self.label_10.setObjectName("label_10")
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(386, 135, 93, 28))
        self.pushButton.setObjectName("pushButton")

        self.label_7 = QtWidgets.QLabel(Form)
        self.label_7.setGeometry(QtCore.QRect(696, 415, 151, 121))
        self.label_7.setObjectName("label_7")
        self.label_6 = QtWidgets.QLabel(Form)
        self.label_6.setGeometry(QtCore.QRect(696, 285, 151, 121))
        self.label_6.setObjectName("label_6")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(60, 40, 261, 211))
        self.label.setObjectName("label")
        self.label_8 = QtWidgets.QLabel(Form)
        self.label_8.setGeometry(QtCore.QRect(216, 415, 151, 121))
        self.label_8.setObjectName("label_8")

        self.retranslateUi(Form)
        self.pushButton.clicked.connect(self.tab1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def tab1(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(directory='.')
        print(fileName)
        pixmap = QPixmap(fileName)
        self.label.setPixmap(pixmap.scaled(self.label.size()))
        self.label.show()
        listResult = query(fileName)
        # print(listResult)
        listResult = ['oxford\\images\\' + s for s in listResult]

        pixmap2 = QPixmap(listResult[0])
        self.label_2.setPixmap(pixmap2.scaled(self.label.size()))
        self.label_2.show()

        pixmap3 = QPixmap(listResult[1])
        self.label_3.setPixmap(pixmap3.scaled(self.label.size()))
        self.label_3.show()

        pixmap4 = QPixmap(listResult[2])
        self.label_4.setPixmap(pixmap4.scaled(self.label.size()))
        self.label_4.show()

        pixmap5 = QPixmap(listResult[3])
        self.label_5.setPixmap(pixmap5.scaled(self.label.size()))
        self.label_5.show()

        pixmap6 = QPixmap(listResult[4])
        self.label_6.setPixmap(pixmap6.scaled(self.label.size()))
        self.label_6.show()

        pixmap7 = QPixmap(listResult[5])
        self.label_7.setPixmap(pixmap7.scaled(self.label.size()))
        self.label_7.show()

        pixmap8 = QPixmap(listResult[6])
        self.label_8.setPixmap(pixmap8.scaled(self.label.size()))
        self.label_8.show()

        pixmap9 = QPixmap(listResult[7])
        self.label_9.setPixmap(pixmap9.scaled(self.label.size()))
        self.label_9.show()

        pixmap10 = QPixmap(listResult[8])
        self.label_10.setPixmap(pixmap10.scaled(self.label.size()))
        self.label_10.show()

        pixmap11 = QPixmap(listResult[9])
        self.label_11.setPixmap(pixmap11.scaled(self.label.size()))
        self.label_11.show()

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "IR"))
        self.label_4.setText(_translate("Form", ""))
        self.label_11.setText(_translate("Form", ""))
        self.label_3.setText(_translate("Form", ""))
        self.label_9.setText(_translate("Form", ""))
        self.label_2.setText(_translate("Form", ""))
        self.label_5.setText(_translate("Form", ""))
        self.label_10.setText(_translate("Form", ""))
        self.pushButton.setText(_translate("Form", "Openfile"))
        self.label_7.setText(_translate("Form", ""))
        self.label_6.setText(_translate("Form", ""))
        self.label.setText(_translate("Form", ""))
        self.label_8.setText(_translate("Form", ""))
        self.label.setStyleSheet("border: 1px solid black;")
        self.label_2.setStyleSheet("border: 1px solid black;")
        self.label_3.setStyleSheet("border: 1px solid black;")
        self.label_4.setStyleSheet("border: 1px solid black;")
        self.label_5.setStyleSheet("border: 1px solid black;")
        self.label_6.setStyleSheet("border: 1px solid black;")
        self.label_7.setStyleSheet("border: 1px solid black;")
        self.label_8.setStyleSheet("border: 1px solid black;")
        self.label_9.setStyleSheet("border: 1px solid black;")
        self.label_10.setStyleSheet("border: 1px solid black;")
        self.label_11.setStyleSheet("border: 1px solid black;")