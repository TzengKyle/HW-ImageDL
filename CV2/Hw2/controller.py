import cv2;
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog

from UI import Ui_MainWindow

def drawContour(img_name):
    img= cv2.imread(img_name)
    (h, w) = img.shape[:2]
    img = cv2.resize(img,(int(w/2), int(h/2)),interpolation=cv2.INTER_AREA)

    edged_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    edged_img = cv2.cvtColor(edged_img,cv2.COLOR_RGB2GRAY)
    edged_img = cv2.GaussianBlur(edged_img, (3, 3), 0)

    edged = cv2.Canny(edged_img, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) # cv2.RETR_EXTERNAL

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
 
    cv2.imshow('Contours' + img_name, img)

def countContour(img_name):
    img= cv2.imread(img_name)
    (h, w) = img.shape[:2]
    img = cv2.resize(img,(int(w/2), int(h/2)),interpolation=cv2.INTER_AREA)

    edged_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    edged_img = cv2.cvtColor(edged_img,cv2.COLOR_RGB2GRAY)
    edged_img = cv2.GaussianBlur(edged_img, (3, 3), 0)

    edged = cv2.Canny(edged_img, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    return int(len(contours)/4)

def Find_Corners(folder_name):
    def drawCorners(img_name):
        img= cv2.imread(img_name)
        (h, w) = img.shape[:2]
        img = cv2.resize(img,(int(w/4), int(h/4)),interpolation=cv2.INTER_AREA)

        ret, cp_img = cv2.findChessboardCorners(img, (11,8), None)
        cv2.drawChessboardCorners(img, (11,8), cp_img, ret)
        return img

    img_list = []
    for i in range(0,15):
        img_name = folder_name + "\\" + str(i+1) + ".bmp"        
        img_list.append(drawCorners(img_name))

    for i in range(0,15):
        cv2.imshow("chessBoard", img_list[i])                # 不斷讀取並顯示串列中的圖片內容
        cv2.waitKey(500)

def Find_Intrinsic(folder_name):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    chess_images = []
    for i in range(0,15):
        img_name = img_name = folder_name + "\\" + str(i+1) + ".bmp"
        chess_images.append(img_name)

    for i in range(len(chess_images)):
        # Read in the image
        image = cv2.imread(chess_images[i])
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)
    # gray.shape[::-1] = (2048, 2048)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
    print("Intrinsic:")
    print(mtx) 

def Find_Extrinsic(folder_name, num):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    chess_images = []
    for i in range(0,15):
        img_name = img_name = folder_name + "\\" + str(i+1) + ".bmp"
        chess_images.append(img_name)

    for i in range(len(chess_images)):
        # Read in the image
        image = cv2.imread(chess_images[i])
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)
    # gray.shape[::-1] = (2048, 2048)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
    
    R = cv2.Rodrigues(rvecs[num-1])
    ext = np.hstack((R[0], tvecs[num-1]))
    print("Extrinsic:")
    print(ext)

def Find_Distortion(folder_name):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    chess_images = []
    for i in range(0,15):
        img_name = img_name = folder_name + "\\" + str(i+1) + ".bmp"
        chess_images.append(img_name)

    for i in range(len(chess_images)):
        # Read in the image
        image = cv2.imread(chess_images[i])
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)
    # gray.shape[::-1] = (2048, 2048)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
    print("Distortion:")
    print(dist) 

def Show_Result(folder_name):
    def origin_img(img_name):
        img= cv2.imread(img_name)
        (h, w) = img.shape[:2]
        img = cv2.resize(img,(int(w/4), int(h/4)),interpolation=cv2.INTER_AREA)
        return img

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    img_list = [] #distort
    for i in range(0,15):
        img_name = folder_name + "\\" + str(i+1) + ".bmp"
        img_list.append(origin_img(img_name))

    chess_images = [] #undistort
    for i in range(0,15):
        img_name = img_name = folder_name + "\\" + str(i+1) + ".bmp"
        chess_images.append(img_name)

    for i in range(len(chess_images)):
        # Read in the image
        image = cv2.imread(chess_images[i])
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)
    # gray.shape[::-1] = (2048, 2048)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
    
    for i in range(0,15):
        img = img_list[i]
        h_img,  w_img = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w_img,h_img), 1, (w_img,h_img)) 
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst,(w_img, h_img),interpolation=cv2.INTER_AREA)

        arrange = np.hstack([img_list[i],dst])
        cv2.imshow("undistorted_chessBoard", arrange)  #不斷讀取並顯示串列中的圖片內容
        cv2.waitKey(500)

def Show_Word_On_Board(folder_name,word):
    char_in_board = [ # coordinate for 6 charter in board (x, y) ==> (w, h)
        [7,5,0], # slot 1
        [4,5,0], # slot 2
        [1,5,0], # slot 3
        [7,2,0], # slot 4
        [4,2,0], # slot 5
        [1,2,0]  # slot 6
    ]

    def drawChar(image, imgpts):
        i = 0
        while i < len(imgpts):
            image = cv2.line(image, tuple(imgpts[i].ravel()), tuple(imgpts[i+1].ravel()), (0, 0, 255), 5)
            i = i + 2
        return image


    fs = cv2.FileStorage('alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)

    word_char = []
    for i in range(len(word)):
        ch = np.float32(fs.getNode(word[i]).mat())

        char_line = []
        for eachline in ch:
            char_line.append(np.add(eachline[0],char_in_board[i])) #np.add(eachline[0],char_in_board[i])
            char_line.append(np.add(eachline[1],char_in_board[i]))
        char_line = np.float32(char_line)

        word_char.append(char_line)
   
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    chess_images = []
    for i in range(0,5):
        img_name = img_name = folder_name + "\\" + str(i+1) + ".bmp"
        chess_images.append(img_name)

    for i in range(0,5):
        image = cv2.imread(chess_images[i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None) #ret is ok

        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
         
            for j in range(len(word_char)):
                imgpts, jac = cv2.projectPoints(word_char[j], rvecs[i], tvecs[i], mtx, dist)
                imgpts = np.int32(imgpts).reshape(-1,2) #關鍵   
                img = drawChar(image, imgpts)

        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', img)
        cv2.waitKey(500)

def Show_Word_Vertically(folder_name,word):
    char_in_board = [ # coordinate for 6 charter in board (x, y) ==> (w, h)
        [7,5,0], # slot 1
        [4,5,0], # slot 2
        [1,5,0], # slot 3
        [7,2,0], # slot 4
        [4,2,0], # slot 5
        [1,2,0]  # slot 6
    ]

    def drawChar(image, imgpts):
        i = 0
        while i < len(imgpts):
            image = cv2.line(image, tuple(imgpts[i].ravel()), tuple(imgpts[i+1].ravel()), (0, 0, 255), 5)
            i = i + 2
        return image


    fs = cv2.FileStorage('alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)

    word_char = []
    for i in range(len(word)):
        ch = np.float32(fs.getNode(word[i]).mat())

        char_line = []
        for eachline in ch:
            char_line.append(np.add(eachline[0],char_in_board[i])) #np.add(eachline[0],char_in_board[i])
            char_line.append(np.add(eachline[1],char_in_board[i]))
        char_line = np.float32(char_line)

        word_char.append(char_line)
   
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((11 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane

    chess_images = []
    for i in range(0,5):
        img_name = img_name = folder_name + "\\" + str(i+1) + ".bmp"
        chess_images.append(img_name)

    for i in range(0,5):
        image = cv2.imread(chess_images[i])
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None) #ret is ok

        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)
            imgpoints.append(corners2)

            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (2048, 2048), None, None)
         
            for j in range(len(word_char)):
                imgpts, jac = cv2.projectPoints(word_char[j], rvecs[i], tvecs[i], mtx, dist)
                imgpts = np.int32(imgpts).reshape(-1,2) #關鍵   
                img = drawChar(image, imgpts)

        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        cv2.imshow('img', img)
        cv2.waitKey(500)

def Stereo_Disparity_Map(imgL_path,imgR_path):
    def draw1(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN :   
            if disparity[y][x] < 0:
                print(disparity[y][x])
                print("failure case")
                return
            print(disparity[y][x])
            disp = disparity[y][x]
            point = (int(x-disp)  , int(y))
            print(x, y, point)
            cv2.imshow('imgR_dot', imgR)
            cv2.circle(imgRC, point, 20, (0,0,255), -1)
            cv2.imshow('imgR_dot', imgRC)
            
    imgL= cv2.imread(imgL_path)
    imgR= cv2.imread(imgR_path)

    (h, w) = imgL.shape[:2]

    imgL_n = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR_n = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    matcher = cv2.StereoBM_create(256,25)
    disparity = matcher.compute(imgL_n, imgR_n)
    disparity = disparity/16
   
    disparity_norm = cv2.normalize(
                disparity,
                disparity, #調整後size
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U,
            )

    imgRC = imgR.copy()

    cv2.namedWindow('imgR_dot', cv2.WINDOW_NORMAL)
    cv2.namedWindow('imgL', cv2.WINDOW_NORMAL)
    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    
    cv2.resizeWindow('imgR_dot', (int(w/5), int(h/5)))
    cv2.resizeWindow('imgL', (int(w/5), int(h/5)))
    cv2.resizeWindow('disparity', (int(w/5), int(h/5)))

    cv2.imshow('imgR_dot', imgR)
    cv2.imshow('imgL', imgL)
    cv2.imshow('disparity', disparity_norm)
    while(1):
        cv2.imshow('imgL', imgL)       
        cv2.setMouseCallback('imgL', draw1)
        imgRC=imgR.copy()
        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()


    def setup_control(self):
        self.folder = "default"
        self.imageL = "default"
        self.imageR = "default"

        self.ui.LoadFolder_Button.clicked.connect(self.click_LoadFolder_Button)
        self.ui.LoadImageL_Button.clicked.connect(self.click_LoadImageL_Button)
        self.ui.LoadImageR_Button.clicked.connect(self.click_LoadImageR_Button)
        
        self.ui.DrawContour_Button.clicked.connect(self.click_DrawContour_Button)
        self.ui.CountRings_Button.clicked.connect(self.click_CountRings_Button)
        
        self.ui.FindCorners_Button.clicked.connect(self.click_FindCorners_Button)
        self.ui.FindIntrinsic_Button.clicked.connect(self.click_FindIntrinsic_Button)
        self.ui.FindExtrinsic_Button.clicked.connect(self.click_FindExtrinsic_Button)
        self.ui.FindDistortion_Button.clicked.connect(self.click_FindDistortion_Button)
        self.ui.ShowResult_Button.clicked.connect(self.click_ShowResult_Button)

        self.ui.ShowWordOnBoard_Button.clicked.connect(self.click_ShowWordOnBoard_Button)
        self.ui.ShowWordVertically_Button.clicked.connect(self.click_ShowWordVertically_Button)

        self.ui.StereoDisparityMap_Button.clicked.connect(self.click_StereoDisparityMap_Button)

    def click_LoadFolder_Button(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.folder = folderpath
        self.ui.LoadFolder_Label.setText(self.folder)

    def click_LoadImageL_Button(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.imageL = filename
        self.ui.LoadImageL_Label.setText(self.imageL)

    def click_LoadImageR_Button(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./")
        self.imageR = filename
        self.ui.LoadImageR_Label.setText(self.imageR)

    def click_DrawContour_Button(self):
        drawContour(self.folder + "\img1.jpg")
        drawContour(self.folder + "\img2.jpg")

    def click_CountRings_Button(self):
        self.ui.RingImg1_Label.setText("There are " + str(countContour(self.folder + "\img1.jpg")) + " rings in img1.jpg")
        self.ui.RingImg2_Label.setText("There are " + str(countContour(self.folder + "\img2.jpg")) + " rings in img2.jpg")   

    def click_FindCorners_Button(self):
        Find_Corners(self.folder)

    def click_FindIntrinsic_Button(self):
        Find_Intrinsic(self.folder)

    def click_FindExtrinsic_Button(self):
        Find_Extrinsic(self.folder, int(self.ui.ChessBoard_ComboBox.currentText()))

    def click_FindDistortion_Button(self):
        Find_Distortion(self.folder)

    def click_ShowResult_Button(self):
        Show_Result(self.folder)

    def click_ShowWordOnBoard_Button(self):
        Show_Word_On_Board(self.folder,self.ui.Word_TextEdit.toPlainText())

    def click_ShowWordVertically_Button(self):
        Show_Word_Vertically(self.folder,self.ui.Word_TextEdit.toPlainText())

    def click_StereoDisparityMap_Button(self):
        Stereo_Disparity_Map(self.imageL,self.imageR)








