import sys
import threading
import numpy as np
from os import path
import os
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
from PyQt5 import QtWidgets
import matplotlib.pyplot as plt
import time
from PyQt5.uic import loadUiType
from PyQt5 import QtCore, QtGui
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.uic import loadUiType
from PyQt5.QtCore import QTimer
from os import path
from scipy import ndimage
import pyqtgraph as pg
import cv2
from matplotlib import pyplot as plt
# from Design import Ui_MainWindow
# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PIL import Image, ImageQt

# img = cv2.imread('kastor.jpeg',0)
# histoImage = cv2.imread('messi5.jpg',0)

FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "Task1_1.ui"))
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self , parent=None):
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        self.title = "Image Filters"
        self.setWindowTitle(self.title)
        self.mode=False
        self.spatialChecked.toggled.connect(self.change_mode)
        self.frequencyChecked.toggled.connect(self.change_mode)
        # pixmap1 = QPixmap('kastor.jpeg')
        # self.OriginalLabel.setPixmap(pixmap1)
        # pixmap2 = QPixmap('messi5.jpg')
        # self.originHistoLabel.setPixmap(pixmap2)
        # self.lowPassButton.clicked.connect( self.lowPass)
        self.browse_photo.triggered.connect(self.browse)
        self.low_pushButton.clicked.connect(lambda:self.multiple_filters(0))
        self.high_pushButton.clicked.connect(lambda:self.multiple_filters(1))
        self.median_pushButton.clicked.connect(lambda:self.multiple_filters(2))
        self.laplace_pushButton.clicked.connect(lambda:self.multiple_filters(3))
        # self.showHisto.clicked.connect(self.HistoGram)
        self.Equalize_Button.clicked.connect(self.hist_equal)
    def browse(self):
        self.file_path_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', " ", "(*.jpg)")
        self.origin_image = cv2.imread(self.file_path_name,0)
        self.original_img_label.setPixmap(QPixmap(self.file_path_name))
        self.HistoGram()
        self.convert_to_freq_domain()
    def change_mode(self):
        if (self.frequencyChecked.isChecked()):
            self.mode=True
            self.laplace_pushButton.setEnabled(False)
            self.median_pushButton.setEnabled(False)
        else:
            self.mode=False
            self.laplace_pushButton.setEnabled(True)
            self.median_pushButton.setEnabled(True)
    def convert_to_freq_domain(self):
        dft = cv2.dft(np.float32(self.origin_image),flags = cv2.DFT_COMPLEX_OUTPUT)
        self.dft_shift = np.fft.fftshift(dft)
        self.magnitude_spectrum = 20*np.log(cv2.magnitude(self.dft_shift[:,:,0],self.dft_shift[:,:,1]))
        cv2.imwrite('original_fourier.jpg',self.magnitude_spectrum)
        self.origin_fourier_label.setPixmap(QPixmap('original_fourier.jpg'))

    def low_high_Pass_freq(self,type):
        rows, cols = self.origin_image.shape
        crow,ccol = int(rows/2) , int(cols/2)
        x=crow
        y=ccol
        self.mask_filter =np.full((rows,cols,2),1,np.uint8)
        if type:
            # mask_filter[int(x-.35*x):int(x+.35*x),int(y-.35*y):int(y+.35*y),:]=0
            for i in range(2*x):
                for j in range(2*y):
                    if (x-i)**2+(y-j)**2<(.25*((400)/2))**2:
                        # print('d5lt',rkm)
                        # rkm +=1
                        #print(type(i))
                        self.mask_filter[i][j]=0
        else:
            # mask_filter =np.full((rows,cols,2),0)
            # mask_filter[int((x-.35*x)):int((x+.5*x)),int((y-.35*y)):int((y+.35*y)),:]=1
            for i in range(2*x):
                for j in range(2*y):
                    if (x-i)**2+(y-j)**2>(.25*((400)/2))**2:
                        # print('d5lt',rkm)
                        # rkm +=1
                        #print(type(i))
                        self.mask_filter[i][j]=0

        self.edited_fourier =  np.multiply(self.dft_shift,self.mask_filter)

        edited_spectrum = cv2.magnitude(self.edited_fourier[:,:,0],self.edited_fourier[:,:,1])
        plt.imshow(edited_spectrum,cmap='gray')
        print('montasaf',self.edited_fourier[x-100,y-100,1])
        plt.show()
        #showing the edited fourier
        self.inverse_fourier()
    def inverse_fourier(self):
        fshift = self.dft_shift*self.mask_filter
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        #print('ay kelma',np.sum(img_back[400,800,1]))
        self.filtered_img = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])


        # f_ishift = np.fft.ifftshift(self.edited_fourier)
        # img_back = cv2.idft(f_ishift)
        # #print('ay kelma',np.sum(img_back[400,800,1]))
        # self.filtered_img = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        #showing the edited photo
        # lowSpatialImage = Image.fromarray(img_back)
        # lowSpatialImage.convert('RGB')
        # lowSpatialImage.save('lowSpatial.png')
        self.show_after_edit()
    def multiple_filters(self,filter_type):
        if self.mode:
            self.low_high_Pass_freq(filter_type)
        elif filter_type<3:
            medianier =(1/9)*np.full((3,3), 1)
            
            high_pass_spatial =(-1/9)*np.full((3,3), 1)
            high_pass_spatial[1,1]=8/9
            
            low_pass_spatial = np.array([[0 ,  1/8 ,  0], [1/8 ,  1/2 ,  1/8], [0 ,  1/8 , 0]])
            
            filters = np.array([low_pass_spatial,high_pass_spatial,medianier])
            applied_filter=filters[filter_type]

            x,y=self.origin_image.shape
            new_img = np.insert(self.origin_image, x-1, self.origin_image[x-1], axis=0)
            new_img = np.insert(new_img, 0, new_img[0], axis=0)
            new_img = np.insert(new_img, y-1, new_img[:,y-1], axis=1)
            new_img = np.insert(new_img, 0, new_img[:,0], axis=1)
            i=1
            j=1
            # print(np.sum(np.multiply(img[i-1:i+2,j-1:j+2],medianier)),img[i-1:i+2,j-1:j+2],medianier)
            filtered_img=np.full((x,y),255)
            for i in range(1,x-1):
                for j in range(1,y-1):
                    # print(i,j)
                    filtered_img[i-1,j-1]=np.sum(np.multiply(self.origin_image[i-1:i+2,j-1:j+2],applied_filter))
            self.filtered_img=filtered_img[0:x-2,0:y-2]
        else:
            print('lapll')
            self.filtered_img = ndimage.laplace(self.origin_image)
        self.show_after_edit()
    def show_after_edit(self):
        # medianImage = Image.fromarray(self.filtered_img)
        # medianImage.save('medianFilter.jpg')
        print('ana b5rg sora')
        cv2.imwrite('editedImage.jpg', self.filtered_img)
        # medianOutput = QPixmap('medianFilter.jpg')
        self.filtered_img_label.setPixmap(QPixmap('editedImage.jpg'))
        
    def HistoGram(self):
        self.unique_elements,self.indices, self.counts_elements = np.unique(self.origin_image,return_inverse=True , return_counts=True)
        self.origin_Histo_Label.plotItem.clearPlots()
        self.origin_Histo_Label.plot(self.unique_elements,self.counts_elements)
        origin_image_pixmap =QPixmap(self.file_path_name)
        origin_image_pixmap.scaledToHeight(self.origin_img_Histo_Label.height())
        # print('height label', self.origin_img_Histo_Label.height())
        origin_image_pixmap.scaledToWidth(self.origin_img_Histo_Label.width())
        self.origin_img_Histo_Label.setPixmap(origin_image_pixmap)
        # self.origin_img_Histo_Label.setPixmap.loadFromData(self.origin_image)

        # print(unique_elements, counts_elements)
        # # print(indices[:15],histoImage[0,:15])
        # for x in range(histoImage.shape[0]):
        #     for y in range(histoImage.shape[1]):
        #         try :
        #             unique_elements_1.index(histoImage[x,y])
        #             #counts_elements_2.append(0)
        #             counts_elements_2[unique_elements_1.index(histoImage[x,y])]+=1
        #             #print('ana f el try')
        #         except:
        #             unique_elements_1.append(histoImage[x,y])
        #             counts_elements_2.append(1)
        #             #print('ana f el except')
        #         # print(unique_elements_1)

        # (unique_elements_1,counts_elements_2,counts_elements_2[i]/sum(counts_elements_2)
    def hist_equal(self):
        pdf=self.counts_elements/np.sum(self.counts_elements)
        summation=0
        cdf=np.array([])
        for prop in pdf:
            summation = prop + summation
            cdf = np.append(cdf,summation)
            # print(summation)
        # print(cdf)
        Sk=255*cdf
        new_points=np.around(Sk)
        self.equalized_histo_label.plotItem.clearPlots()
        self.equalized_histo_label.plot(new_points,self.counts_elements)
        # plt.plot(new_points)
        # print(len(unique_elements_1) ,len(cdf),new_points)
        new_histoImage = self.indices
        for index_of_unique,unique in enumerate(self.unique_elements) :
            for index_of_pixel,pixel in enumerate(self.indices):
                if pixel == unique:
                    new_histoImage[index_of_pixel] = new_points[index_of_unique]
        # plt.bar(new_points,counts_elements_2)
        # print(len(new_histoImage), len(counts_elements_2))
        new_histoImage=np.reshape(new_histoImage,self.origin_image.shape)
        print(new_histoImage.shape)
        # histoGramImage = Image.fromarray(new_histoImage)
        # new_histoImage.save('EqualizedImage.jpg')
        cv2.imwrite('EqualizedImage.jpg', new_histoImage)
        HistoOutput = QPixmap('EqualizedImage.jpg')
        HistoOutput.scaledToHeight(self.equalized_img_Histo_Label.height())
        HistoOutput.scaledToWidth(self.equalized_img_Histo_Label.width())
        self.equalized_img_Histo_Label.setPixmap(HistoOutput)



    
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()
