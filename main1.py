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
import time




# img = cv2.imread('kastor.jpeg',0)
# histoImage = cv2.imread('messi5.jpg',0)

FORM_CLASS,_ = loadUiType(path.join(path.dirname(__file__), "Task1_1.ui"))
class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self , parent=None):
        pg.setConfigOption('background', (25,25,35))
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        
        self.title = "Image Filters"
        self.setWindowTitle(self.title)
        self.setWindowIcon(QIcon("images/icons/wizard.png"))
        self.mode=False
        self.spatialChecked.toggled.connect(self.change_mode)
        self.frequencyChecked.toggled.connect(self.change_mode)
        # pixmap1 = QPixmap('kastor.jpeg')
        # self.OriginalLabel.setPixmap(pixmap1)
        # pixmap2 = QPixmap('messi5.jpg')
        # self.originHistoLabel.setPixmap(pixmap2)
        # self.lowPassButton.clicked.connect( self.lowPass)
        self.browse_photo.triggered.connect(self.browse)
        self.actionClear_All.triggered.connect(self.clear_frames)
        self.low_pushButton.clicked.connect(lambda:self.multiple_filters(0))
        self.high_pushButton.clicked.connect(lambda:self.multiple_filters(1))
        self.median_pushButton.clicked.connect(lambda:self.multiple_filters(2))
        self.laplace_pushButton.clicked.connect(lambda:self.multiple_filters(3))
        # self.showHisto.clicked.connect(self.HistoGram)
        self.Equalize_Button.clicked.connect(self.hist_equal)
    def browse(self):
        self.file_path_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', " ", "JPG Files (*.jpg; *.jpeg);; PNG Fiels (*.png);; BMP Files (*.bmp);; All Files (*)")
        self.clear_frames()
        self.origin_image = cv2.imread(self.file_path_name,0)
        self.original_img_label.setPixmap(QPixmap(self.file_path_name))
        self.HistoGram()
        self.convert_to_freq_domain()
            
    ######## Clears all labels and graphicsViews####### 
    
    def clear_frames(self):
        labels = [self.origin_Histo_Label, self.equalized_histo_label,
                  self.origin_img_Histo_Label, self.equalized_img_Histo_Label, 
                  self.original_img_label, self.filtered_img_label,
                  self.origin_fourier_label, self.filtered_fourier_label]
        for n in labels:
            n.clear()
     
    ###############################################

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
                    if (x-i)**2+(y-j)**2<int((1/6)*(x+y))**2:
                        # print('d5lt',rkm)
                        # rkm +=1
                        #print(type(i))
                        self.mask_filter[i][j]=0
        else:
            # mask_filter =np.full((rows,cols,2),0)
            # mask_filter[int((x-.35*x)):int((x+.5*x)),int((y-.35*y)):int((y+.35*y)),:]=1
            for i in range(2*x):
                for j in range(2*y):
                    if (x-i)**2+(y-j)**2>int((1/6)*(x+y))**2:
                        # print('d5lt',rkm)
                        # rkm +=1
                        #print(type(i))
                        self.mask_filter[i][j]=0

        self.edited_fourier =  np.multiply(self.dft_shift,self.mask_filter)

        edited_spectrum = np.abs(self.edited_fourier[:,:,0],self.edited_fourier[:,:,1])
        # plt.imshow(edited_spectrum,cmap='gray')
        print('montasaf',self.edited_fourier[x-100,y-100,1])
        # plt.show()
        #showing the edited fourier
        self.inverse_fourier()
    def inverse_fourier(self):
        fshift = self.dft_shift*self.mask_filter
        # f_ishift = np.fft.ifftshift(fshift)
        # img_back = cv2.idft(f_ishift)
        # #print('ay kelma',np.sum(img_back[400,800,1]))
        # self.filtered_img = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        # f_ishift = np.fft.ifftshift(fshift)
        # img_back = np.fft.ifft(f_ishift)
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        self.filtered_img = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        # self.filtered_img = np.real(img_back)
        # plt.imshow(self.filtered_img,cmap='gray')
        # plt.show()
        # f_ishift = np.fft.ifftshift(self.edited_fourier)
        # img_back = cv2.idft(f_ishift)
        # #print('ay kelma',np.sum(img_back[400,800,1]))
        # self.filtered_img = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        #showing the edited photo
        # lowSpatialImage = Image.fromarray(img_back)
        # lowSpatialImage.convert('RGB')
        # lowSpatialImage.save('lowSpatial.png')
        self.show_after_edit()
        self.fourier_after()
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
            # print('lapll')
            self.filtered_img = ndimage.laplace(self.origin_image)
            print(self.filtered_img.shape)
        self.show_after_edit()
        self.fourier_after()
    def show_after_edit(self):
        # medianImage = Image.fromarray(self.filtered_img)
        # medianImage.save('medianFilter.jpg')
        # print('ana b5rg sora')
        plt.imsave('editedImage.jpg', self.filtered_img,cmap='gray')
        # medianOutput = QPixmap('medianFilter.jpg')
        self.filtered_img_label.setPixmap(QPixmap('editedImage.jpg'))
    def fourier_after(self):
        dft_filtered = cv2.dft(np.float32(self.filtered_img),flags = cv2.DFT_COMPLEX_OUTPUT)
        self.dft_shift_filtered = np.fft.fftshift(dft_filtered)
        self.magnitude_spectrum_filtered = 20*np.log(cv2.magnitude(self.dft_shift_filtered[:,:,0],self.dft_shift_filtered[:,:,1]))
        plt.imsave('filtered_fourier.jpg',self.magnitude_spectrum_filtered,cmap='gray')
        self.filtered_fourier_label.setPixmap(QPixmap('filtered_fourier.jpg'))

    def HistoGram(self):
        self.unique_elements,self.indices, self.counts_elements = np.unique(self.origin_image,return_inverse=True , return_counts=True)
        self.origin_Histo_Label.plotItem.clearPlots()
        self.origin_Histo_Label.plot(self.unique_elements,self.counts_elements, pen='c')
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
        self.start_time = time.time()
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
        self.equalized_histo_label.plot(new_points,self.counts_elements, pen='c')
        # plt.plot(new_points)
        # print(len(unique_elements_1) ,len(cdf),new_points)
        
        # new_histoImage = self.indices
        # for index_of_unique,unique in enumerate(self.unique_elements) :
        #     for index_of_pixel,pixel in enumerate(self.indices):
        #         if pixel == unique:
        #             new_histoImage[index_of_pixel] = new_points[index_of_unique]
                    
        self.indices = np.array(self.indices)
        new_histoImage = np.full(len(self.indices), 0)
        for index_of_unique,unique in enumerate(self.unique_elements):
            new_histoImage[self.indices==unique] = new_points[index_of_unique]
                    
                    
        # plt.bar(new_points,counts_elements_2)
        # print(len(new_histoImage), len(counts_elements_2))
        new_histoImage=np.reshape(new_histoImage,self.origin_image.shape)
        
        # unique2, count2 = np.unique(new_histoImage, return_counts=True)
        # self.equalized_histo_label.plotItem.clearPlots()
        # self.equalized_histo_label.plot(count2, pen='c')
        
        # print(new_histoImage.shape)
        # histoGramImage = Image.fromarray(new_histoImage)
        # new_histoImage.save('EqualizedImage.jpg')
        cv2.imwrite('EqualizedImage.jpg', new_histoImage)
        HistoOutput = QPixmap('EqualizedImage.jpg')
        HistoOutput.scaledToHeight(self.equalized_img_Histo_Label.height())
        HistoOutput.scaledToWidth(self.equalized_img_Histo_Label.width())
        self.equalized_img_Histo_Label.setPixmap(HistoOutput)
        print("--- %s seconds ---" % (time.time() - self.start_time))
        
    def hist_equal_os(self):
        img_gray = cv2.imread(self.file_path_name, cv2.IMREAD_GRAYSCALE)
        unique2, count2 = np.unique(img_gray, return_counts=True)
        c_sum = np.cumsum(count2)
        N = c_sum.max() - c_sum.min()
        n = c_sum - c_sum.min()
        n *= 255 
        s = n/N
        s = s.astype('uint8')
        img_flat = img_gray.flatten()
        new_img = np.full(len(img_flat), 0)
        
        
        img_flat = np.array(img_flat)
        for hist_index, hist_value in enumerate(s):
                new_img[img_flat == hist_index] = hist_value
                    
        # for index, value in enumerate(img_flat):
        #     for hist_index, hist_value in enumerate(s):
        #         if value == hist_index:
        #             new_img[index] = hist_value
                    
                    
        new_img = np.reshape(new_img, img_gray.shape)
        unique_new, count_new = np.unique(new_img, return_counts=True)
        self.equalized_histo_label.plotItem.clearPlots()
        self.equalized_histo_label.plot(unique_new, count_new, pen='c')
        
        cv2.imwrite('EqualizedImage.jpg', new_img)
        HistoOutput = QPixmap('EqualizedImage.jpg')
        HistoOutput.scaledToHeight(self.equalized_img_Histo_Label.height())
        HistoOutput.scaledToWidth(self.equalized_img_Histo_Label.width())
        self.equalized_img_Histo_Label.setPixmap(HistoOutput)
        

    
def main():
    app = QApplication(sys.argv)
    style = """
        QWidget{
            color: white;
            background: #262D37;
            font-weight: bold;
            font-size: 12px;
            }
        QGraphicsView{
            border: 1px solid #fff;
            border-radius: 4px;
            padding: 2px;
            color: #fff;
            }
        QPushButton{
            color: white;
            background: #1C658C;
            border: 1px #DADADA solid;
            padding: 4px 10px;
            border-radius:  2px;
            font-weight: bold;
            font-size: 12px;
            outline: none;
            }
        QPushButton:hover{
            border: 1px #C6C6C6 solid;
            background: #0892D0;
            }
        QPushButton:!enabled{
            border: 1px #C6C6C6 solid;
            background: #88a8b9;
            }
        QTabWidget::pane {
            color: black;
            background: #fff;
            border: 1px #DADADA solid;
            padding: 1px 1px;
            border-radius: 4px;
            }
        QTabBar::tab {
            background: #262D37; 
            border: 1px solid lightgray; 
            border-radius: 4px;
            padding: 5px;
            } 
        QTabBar::tab:selected { 
            background: #4d5b70; 
            margin-bottom: -1px; 
            }
        QLabel{
            border: 1px solid #fff;
            border-radius: 4px;
            padding: 2px;
            color: #fff;
            }
        QGroupBox{
            border: 1px solid #fff;
            padding: 4px 10px;
            border-radius:  4px;
            }
        QMenuBar{
            /* background: #262D37; */
            }
        QMenuBar{
            /* border: 1px solid #fff;
            padding: 4px 10px;
            */
            border-radius: 4px;
            }
        QMenuBar::item::selected{
            background: #4b596b;
            color: #fff;
            }
        QMenu::item{
            /* background: #3b444f;
            border: 1px #C6C6C6 solid;
            border-radius: 4px; */
            }
        QMenu::item::selected
            {
            background: #4b596b;
            color: #fff;
            }
    """
    app.setStyleSheet(style)
    window = MainApp()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()
