import os
import cv2
import cv2 as cv
import sys
import time
import threading
import numpy as np
from os import path
import matplotlib.pyplot as plt
from PyQt5.uic import loadUiType
from PyQt5 import QtCore, QtGui
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer
from scipy import ndimage
import pyqtgraph as pg
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
        pg.setConfigOption('background', (25,25,35))
        # pg.setConfigOption('foreground', (255,215,0))
        super(MainApp,self).__init__(parent)
        QMainWindow.__init__(self)
        self.setupUi(self)
        
        if not os.path.exists('cache'):
            os.makedirs('cache')
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
        self.Equalize_Button.clicked.connect(self.hist_equal_os)
        
    def browse(self):
        self.file_path_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open file', " ", "JPG Files (*.jpg; *.jpeg);; PNG Fiels (*.png);; BMP Files (*.bmp);; All Files (*)")
        self.clear_frames()
        self.origin_image = cv2.imread(self.file_path_name,0)
        self.origin_image_bgr = cv2.imread(self.file_path_name)
        self.origin_image_rgb = cv2.cvtColor(self.origin_image_bgr, cv2.COLOR_BGR2RGB)
        self.original_img_label.setPixmap(QPixmap(self.file_path_name))
        cv.imwrite('cache/origin_gray.jpg', self.origin_image)
        self.origin_gray_pixmap =QPixmap('cache/origin_gray.jpg')
        self.HistoGram()
        self.convert_to_freq_domain()
            
    ######## Clears all labels and graphicsViews####### 
    
    def clear_frames(self):
        labels = [self.origin_histo_graph, self.equalized_histo_graph,
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
        cv2.imwrite('cache/original_fourier.jpg',self.magnitude_spectrum)
        self.origin_fourier_label.setPixmap(QPixmap('cache/original_fourier.jpg'))

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
        self.filter_type = filter_type
        self.original_img_label_2.setPixmap(QPixmap(self.file_path_name))
        self.Picking_Image_Spatial()
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
            if filter_type<2:
                for i in range(1,x-1):
                    for j in range(1,y-1):
                        # print(i,j)
                            filtered_img[i-1,j-1]=np.sum(np.multiply(self.origin_image[i-1:i+2,j-1:j+2],applied_filter))
            else:
                for i in range(1,x-1):
                    for j in range(1,y-1):
                        filtered_img[i-1,j-1]=np.median(self.origin_image[i-1:i+2,j-1:j+2])
            self.filtered_img=filtered_img[0:x-2,0:y-2]
        else:

            # print('lapll')
            self.filtered_img = cv2.Laplacian(self.origin_image, cv2.CV_16S, ksize=3)
            self.filtered_img = cv2.convertScaleAbs(self.filtered_img)
            # self.filtered_img = ndimage.laplace(self.origin_image)
            print(self.filtered_img.shape)
        self.show_after_edit()
        self.fourier_after()
    def show_after_edit(self):
        # medianImage = Image.fromarray(self.filtered_img)
        # medianImage.save('medianFilter.jpg')
        # print('ana b5rg sora')
        plt.imsave('cache/editedImage.jpg', self.filtered_img,cmap='gray')
        # medianOutput = QPixmap('medianFilter.jpg')
        self.filtered_img_label.setPixmap(QPixmap('cache/editedImage.jpg'))
    def fourier_after(self):
        dft_filtered = cv2.dft(np.float32(self.filtered_img),flags = cv2.DFT_COMPLEX_OUTPUT)
        self.dft_shift_filtered = np.fft.fftshift(dft_filtered)
        self.magnitude_spectrum_filtered = 20*np.log(cv2.magnitude(self.dft_shift_filtered[:,:,0],self.dft_shift_filtered[:,:,1]))
        plt.imsave('cache/filtered_fourier.jpg',self.magnitude_spectrum_filtered,cmap='gray')
        self.filtered_fourier_label.setPixmap(QPixmap('cache/filtered_fourier.jpg'))

    def HistoGram(self):
        self.unique_elements,self.indices, self.counts_elements = np.unique(self.origin_image,return_inverse=True , return_counts=True)
        self.origin_histo_graph.plotItem.clearPlots()
        self.origin_histo_graph.plot(self.unique_elements,self.counts_elements, pen='c')
        self.origin_image_pixmap =QPixmap(self.file_path_name)
        self.origin_image_pixmap.scaledToHeight(self.origin_img_Histo_Label.height())
        # print('height label', self.origin_img_Histo_Label.height())
        self.origin_image_pixmap.scaledToWidth(self.origin_img_Histo_Label.width())
        if (self.checkBox_colored.isChecked()):
            self.origin_img_Histo_Label.setPixmap(self.origin_image_pixmap)
        else:
            self.origin_img_Histo_Label.setPixmap(self.origin_gray_pixmap)
            
            
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
        self.equalized_histo_graph.plotItem.clearPlots()
        self.equalized_histo_graph.plot(new_points,self.counts_elements, pen='c')
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
        # self.equalized_histo_graph.plotItem.clearPlots()
        # self.equalized_histo_graph.plot(count2, pen='c')
        
        # print(new_histoImage.shape)
        # histoGramImage = Image.fromarray(new_histoImage)
        # new_histoImage.save('EqualizedImage.jpg')
        cv2.imwrite('cache/EqualizedImage.jpg', new_histoImage)
        HistoOutput = QPixmap('cache/EqualizedImage.jpg')
        HistoOutput.scaledToHeight(self.equalized_img_Histo_Label.height())
        HistoOutput.scaledToWidth(self.equalized_img_Histo_Label.width())
        self.equalized_img_Histo_Label.setPixmap(HistoOutput)
        print("--- %s seconds ---" % (time.time() - self.start_time))
        
    def hist_equal_os(self):
        img_gray = cv2.imread(self.file_path_name, cv2.IMREAD_GRAYSCALE)
        
        # img = cv2.imread(self.file_path_name)
        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # hsvrgb = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        # img_gray = hsv[::2]
        
        # cv2.imshow('Original image',img)
        # cv2.imshow('HSV image', hsv)
        # cv2.imshow('FromHSVtoRGB image', hsvrgb)
        
        unique2, count2 = np.unique(img_gray, return_counts=True) #misses values with 0 pixels
        # print(unique2)
        # plt.bar(unique2, count2)
        count_full = np.full(256, 0)
        count_full[:len(count2)] += count2
        w,h = img_gray.shape
        img_size = w*h
        
        img_flat = img_gray.flatten()
        count_full = np.zeros(256)
        
        for i in range (1,img_size): 
            count_full[img_flat[i]] += 1

        P = count_full/img_size
        c_sum = np.cumsum(P)
        c_sum *= 255
        plt.plot(c_sum)
        c_sum = np.round(c_sum)
        # N = c_sum.max() - c_sum.min()
        # n = c_sum - c_sum.min()
        # n *= 255 
        # s = n/N
        # s = s.astype('uint8')
        new_img = np.full(len(img_flat), 0)
        # new_img = np.zeros_like(img_flat)
        
        img_flat = np.array(img_flat)
        for hist_index, hist_value in enumerate(c_sum):
                new_img[img_flat == hist_index] = hist_value
                    
        # for index, value in enumerate(img_flat):
        #     for hist_index, hist_value in enumerate(s):
        #         if value == hist_index:
        #             new_img[index] = hist_value
                    
        new_img = np.reshape(new_img, img_gray.shape)
        unique_new, count_new = np.unique(new_img, return_counts=True)
        
        ################ Colored Histogram #################
        if(self.checkBox_colored.isChecked()):
            self.origin_img_Histo_Label.setPixmap(self.origin_image_pixmap)
            # normalized_intens = new_img / 255
            bgr = cv2.imread(self.file_path_name)
            hsi = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            print('hsi[2]', hsi[:,:,2])
            hsi[:,:,2] = new_img
            # OpenCV stores RGB values inverting R and B channels, i.e. BGR.
            new_img = cv2.cvtColor(hsi, cv2.COLOR_HSV2BGR)
        else:
            self.origin_img_Histo_Label.setPixmap(self.origin_gray_pixmap)
        ####################################################
        
        self.equalized_histo_graph.plotItem.clearPlots()
        self.equalized_histo_graph.plot(unique_new, count_new, pen='c')
        # self.equalized_histo_graph.plot(unique2, count2, pen='c')
        cv2.imwrite('cache/EqualizedImage.jpg', new_img)
        HistoOutput = QPixmap('cache/EqualizedImage.jpg')
        # HistoOutput.scaledToHeight(self.equalized_img_Histo_Label.height())
        # HistoOutput.scaledToWidth(self.equalized_img_Histo_Label.width())
        self.equalized_img_Histo_Label.setPixmap(HistoOutput)
        
        # cv2.imshow("new image",new_img)
        # cv2.waitKey(0)

    def Picking_Image_Spatial(self): 
        self.image_spatial = self.origin_image_bgr
        
        
        if(self.filter_type == 0):
            self.blur = cv.GaussianBlur(self.image_spatial,(5,5),cv.BORDER_DEFAULT)
            
            cv.imwrite("cache/blur.jpg",self.blur)
            self.rgb_img = "cache/blur.jpg"
            
            self.rgb_img_label.setPixmap(QPixmap(self.rgb_img))
            self.blur_gray = cv2.cvtColor(self.blur, cv2.COLOR_BGR2GRAY)
            
            cv.imwrite("cache/gray_blur.jpg",self.blur_gray)
            self.gray_img = "cache/gray_blur.jpg"
            
            self.gray_img_label.setPixmap(QPixmap(self.gray_img))
            
            cv.imwrite("cache/gray_origin.jpg",self.origin_image)
            self.gray_origin_img = "cache/gray_origin.jpg"
            
            self.original_gray_img_label.setPixmap(QPixmap(self.gray_origin_img))
            
        elif(self.filter_type == 1):
            self.edges = self.image_spatial - cv2.GaussianBlur(self.image_spatial, (21, 21), 3)+127 # cv.GaussianBlur(self.image_spatial,(3,3),cv.BORDER_DEFAULT)
            # self.edges = cv.Canny(self.edge,100,150)
            
            cv.imwrite("cache/edges.jpg",self.edges)
            self.rgb_img = "cache/edges.jpg"
            
            self.rgb_img_label.setPixmap(QPixmap(self.rgb_img))
            self.edge_gray = cv2.cvtColor(self.edges, cv2.COLOR_BGR2GRAY)
            
            cv.imwrite("cache/edge_gray.jpg",self.edge_gray)
            self.gray_img = "cache/edge_gray.jpg"
            
            self.gray_img_label.setPixmap(QPixmap(self.gray_img))
            
            cv.imwrite("cache/gray_origin.jpg",self.origin_image)
            self.gray_origin_img = "cache/gray_origin.jpg"
            
            self.original_gray_img_label.setPixmap(QPixmap(self.gray_origin_img))
            
        elif(self.filter_type == 2):
            self.blur_2 = cv.medianBlur(self.image_spatial,7)
            
            cv.imwrite("cache/blur_2.jpg",self.blur_2)
            self.rgb_img = "cache/blur_2.jpg"
            
            self.rgb_img_label.setPixmap(QPixmap(self.rgb_img))
            self.blur2_gray = cv2.cvtColor(self.blur_2, cv2.COLOR_BGR2GRAY)
            
            cv.imwrite("cache/blur2_gray.jpg",self.blur2_gray)
            self.gray_img = "cache/blur2_gray.jpg"
            
            self.gray_img_label.setPixmap(QPixmap(self.gray_img))
            
            cv.imwrite("cache/gray_origin.jpg",self.origin_image)
            self.gray_origin_img = "cache/gray_origin.jpg"
            
            self.original_gray_img_label.setPixmap(QPixmap(self.gray_origin_img))

        elif(self.filter_type == 3):
            self.kernel2 = np.matrix('-1 -1 -1;-1 8 -1;-1 -1 -1', np.float64)
            self.Laplacian = cv.filter2D(src=self.image_spatial, ddepth=-1, kernel=self.kernel2)
            
            cv.imwrite("cache/Laplaciann.jpg",self.Laplacian)
            self.rgb_img = "cache/Laplaciann.jpg"
            
            self.rgb_img_label.setPixmap(QPixmap(self.rgb_img))
            self.lap_gray = cv2.cvtColor(self.Laplacian, cv2.COLOR_BGR2GRAY)
            
            cv.imwrite("cache/lap_gray.jpg",self.lap_gray)
            self.gray_img = "cache/lap_gray.jpg"
            
            self.gray_img_label.setPixmap(QPixmap(self.gray_img))
            
            cv.imwrite("cache/gray_origin.jpg",self.origin_image)
            self.gray_origin_img = "cache/gray_origin.jpg"
            
            self.original_gray_img_label.setPixmap(QPixmap(self.gray_origin_img))

    
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
            padding: 0px;
            min-width: 90px;
            min-height: 25px;
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
        QMenu{
            /*border: 1px solid #fff;   
            min-width: 100px;
            min-height: 70px;*/
            background-color: #445060;
            color: rgb(255,255,255);
            }
        QMenu::item{
            /* background: #3b444f;
            border: 1px #C6C6C6 solid;
            border-radius: 4px; */
            }
        QMenu::item::selected
            {
            background: #5F7289;
            color: #fff;
            }
    """
    app.setStyleSheet(style)
    window = MainApp()
    window.show()
    app.exec_()
if __name__ == '__main__':
    main()
