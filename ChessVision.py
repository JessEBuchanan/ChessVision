import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from keras_preprocessing import image
from scikeras.wrappers import KerasClassifier, KerasRegressor
import cv2
import glob
from pyforms import start_app
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 

class chessSquare:
     
    def __init__(self, name, startRow, endRow, startCol, endCol): #name is the square name, space is the subarray range within the chessboard
        squares ={ 
            'a1' : 'black',
            'a2' : 'white',
            'a3' : 'black',
            'a4' : 'white',
            'a5' : 'black',
            'a6' : 'white',
            'a7' : 'black',
            'a8' : 'white',
            'b1' : 'white',
            'b2' : 'black',
            'b3' : 'white',
            'b4' : 'black',
            'b5' : 'white',
            'b6' : 'black',
            'b7' : 'white',
            'b8' : 'black',
            'c1' : 'black',
            'c2' : 'white',
            'c3' : 'black',
            'c4' : 'white',
            'c5' : 'black',
            'c6' : 'white',
            'c7' : 'black',
            'c8' : 'white',
            'd1' : 'white',
            'd2' : 'black',
            'd3' : 'white',
            'd4' : 'black',
            'd5' : 'white',
            'd6' : 'black',
            'd7' : 'white',
            'd8' : 'black',
            'e1' : 'black',
            'e2' : 'white',
            'e3' : 'black',
            'e4' : 'white',
            'e5' : 'black',
            'e6' : 'white',
            'e7' : 'black',
            'e8' : 'white',
            'f1' : 'white',
            'f2' : 'black',
            'f3' : 'white',
            'f4' : 'black',
            'f5' : 'white',
            'f6' : 'black',
            'f7' : 'white',
            'f8' : 'black',
            'g1' : 'black',
            'g2' : 'white',
            'g3' : 'black',
            'g4' : 'white',
            'g5' : 'black',
            'g6' : 'white',
            'g7' : 'black',
            'g8' : 'white',
            'h1' : 'white',
            'h2' : 'black',
            'h3' : 'white',
            'h4' : 'black',
            'h5' : 'white',
            'h6' : 'black',
            'h7' : 'white',
            'h8' : 'black',
            }
        self.name = name
        self.StartRow = startRow #may not need to adjust these numbers
        self.EndRow = endRow
        self.StartCol = startCol
        self.EndCol = endCol
        self.color = squares[name] #probably don't need this
        self.occupied = 0

class chessPiece: 
    def __init__(self, name,color, start):
        self.name = name
        self.color = color
        self.current = start
        
def getBoardDim(image): #determines approximate dimensions of each square
    boardDim = image.shape[0]
    boardDim = boardDim//8
    return boardDim

def newGame(board, boardDim): #uses dimensions of the board image to partition into the individual spaces. Create chesspiece objects and set their starting current position. 
    spaces = np.zeros((8,8), dtype=chessSquare)   
    files = ['h', 'g', 'f', 'e', 'd','c','b','a']
    ranks = ['1','2','3','4','5','6','7','8']

    for i in range(0,8):
        for j in range(0,8):
            spaces[i][j]=chessSquare(files[j]+ranks[i],(boardDim*i),(boardDim*(i+1)),(boardDim*j),(boardDim*(j+1)))
 
    
    blackPieces = []
    whitePieces = []

    blackPieces.append(chessPiece('blackLightRook','black',spaces[7][7].name))
    blackPieces.append(chessPiece('blackDarkKnight', 'black', spaces[7][6].name))
    blackPieces.append(chessPiece('blackLightBishop', 'black', spaces[7][5].name))
    blackPieces.append(chessPiece('blackQueen', 'black', spaces[7][4].name))
    blackPieces.append(chessPiece('blackKing', 'black', spaces[7][3].name))
    blackPieces.append(chessPiece('blackDarkBishop', 'black', spaces[7][2].name))
    blackPieces.append(chessPiece('blackLightKnight', 'black', spaces[7][1].name))
    blackPieces.append(chessPiece('blackDarkRook', 'black', spaces[7][0].name))
    blackPieces.append(chessPiece('blackAPawn', 'black', spaces[6][7].name))
    blackPieces.append(chessPiece('blackBPawn', 'black', spaces[6][6].name))
    blackPieces.append(chessPiece('blackCPawn', 'black', spaces[6][5].name))
    blackPieces.append(chessPiece('blackDPawn', 'black', spaces[6][4].name))
    blackPieces.append(chessPiece('blackEPawn', 'black', spaces[6][3].name))
    blackPieces.append(chessPiece('blackFPawn', 'black', spaces[6][2].name))
    blackPieces.append(chessPiece('blackGPawn', 'black', spaces[6][1].name))
    blackPieces.append(chessPiece('blackHPawn', 'black', spaces[6][0].name))

    whitePieces.append(chessPiece('whiteDarkRook','white', spaces[0][7].name))
    whitePieces.append(chessPiece('whiteLightKnight', 'white', spaces[0][6].name))
    whitePieces.append(chessPiece('whiteDarkBishop', 'white', spaces[0][5].name))
    whitePieces.append(chessPiece('whiteQueen', 'white', spaces[0][4].name))
    whitePieces.append(chessPiece('whiteKing', 'white', spaces[0][3].name))
    whitePieces.append(chessPiece('whiteLightBishop', 'white', spaces[0][2].name))
    whitePieces.append(chessPiece('whiteDarkKnight', 'white', spaces[0][1].name))
    whitePieces.append(chessPiece('whiteLightRook', 'white', spaces[0][0].name))
    whitePieces.append(chessPiece('whiteAPawn', 'white', spaces[1][7].name))
    whitePieces.append(chessPiece('whiteBPawn', 'white', spaces[1][6].name))
    whitePieces.append(chessPiece('whiteCPawn', 'white', spaces[1][5].name))
    whitePieces.append(chessPiece('whiteDPawn', 'white', spaces[1][4].name))
    whitePieces.append(chessPiece('whiteEPawn','white', spaces[1][3].name))
    whitePieces.append(chessPiece('whiteFPawn', 'white', spaces[1][2].name))
    whitePieces.append(chessPiece('whiteGPawn', 'white', spaces[1][1].name))
    whitePieces.append(chessPiece('whiteHPawn', 'white', spaces[1][0].name))

    return spaces, blackPieces, whitePieces

def checkOccupied(chessSpace, image, dim):
    space = image[chessSpace.StartRow:chessSpace.EndRow, chessSpace.StartCol:chessSpace.EndCol]
    imageResize = cv2.resize(space, (dim,dim), cv2.INTER_LINEAR)
    #gray = cv2.cvtColor(imageResize, cv2.COLOR_BGR2GRAY)
    grayArray = np.zeros((1,dim,dim,3))
    #grayArray = grayArray
    grayArray[0] = imageResize
    y_pred = model.predict(grayArray)
    #print(y_pred)
    pred = np.argmax(y_pred[0])
    chessSpace.occupied = pred

class staticROI(object):
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False
        self.croppedPlay = [] 

        self.update()

    def update(self):
        while True:
            if self.capture.isOpened():
                # Read frame
                (self.status, self.frame) = self.capture.read()
                cv2.imshow('Play Area', self.frame)
                key = cv2.waitKey(2)

                # Crop image
                if key == ord('c'):
                    self.clone = self.frame.copy()
                    cv2.namedWindow('image')
                    cv2.setMouseCallback('image', self.extract_coordinates)
                    while True:
                        key = cv2.waitKey(2)
                        cv2.imshow('image', self.clone)

                        # Crop and display cropped image
                        if key == ord('c'):
                            self.crop_ROI()
                            self.show_cropped_ROI()

                        # Resume video
                        if key == ord('r'):
                            self.capture.release()
                            cv2.destroyAllWindows()
                            return
                # Close program with keyboard 'q'
                if key == ord('q'):
                    self.capture.release()
                    cv2.destroyAllWindows()
                    return
            else:
                pass

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x,y)]
            self.extract = True

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x,y))
            self.extract = False

            self.selected_ROI = True

            # Draw rectangle around ROI
            cv2.rectangle(self.clone, self.image_coordinates[0], self.image_coordinates[1], (0,255,0), 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.frame.copy()
            self.selected_ROI = False

    def crop_ROI(self):
        if self.selected_ROI:
            self.cropped_image = self.frame.copy()

            x1 = self.image_coordinates[0][0]
            y1 = self.image_coordinates[0][1]
            x2 = self.image_coordinates[1][0]
            y2 = self.image_coordinates[1][1]

            self.cropped_image = self.cropped_image[y1:y2, x1:x2]
            self.croppedPlay = self.cropped_image

            print('Cropped play area: {} {}'.format(self.image_coordinates[0], self.image_coordinates[1]))
            
        else:
            print('Select play area to crop before cropping')

    def show_cropped_ROI(self):
        cv2.imshow('Cropped play area', self.cropped_image)

dim = 100
size = (dim,dim)
board = cv2.imread('C:/Users/Jess/Documents/ChessVision/Training/Advanced Variation/start.jpg')
boardDim = getBoardDim(board)
boardSpaces = np.zeros((8,8)) 
blackPieces = []
whitePieces = []

# gui
from pyforms.basewidget import BaseWidget
from pyforms.controls   import ControlFile
from pyforms.controls   import ControlText
from pyforms.controls   import ControlButton
from pyforms.controls   import ControlLabel

class ComputerVisionAlgorithm(BaseWidget):

    def __init__(self, *args, **kwargs):
        super().__init__('ChessVision')

        #Definition of the forms fields
        self._newgamebutton = ControlButton('New Game')
        self._movebutton     = ControlButton('Move')
        self._outputlabel = ControlLabel('To set up a new game, after you click New Game, your webcam will open. Position over the chessboard and press c. A duplicate image will pop up - drag your mouse to make a box over the 8x8 playing area. When finished, press c and then r.')

        
        #Define the event that will be called when the move button is processed
        self._movebutton.value       = self.__moveEvent
        #Define the event that will be called when the new game button is processed
        self._newgamebutton.value       = self.__newgamebuttonEvent
        #Define the organization of the Form Controls
        self._formset = [
            ('_outputlabel'),
            ('_newgamebutton'),
            '_movebutton'
            
        ]
    
    
    def __newgamebuttonEvent(self):
        newBoard = staticROI()
        board = newBoard.croppedPlay
        print(board.shape)
        boardDim = getBoardDim(board)
        boardSpaces, blackPieces, whitePieces = newGame(board, boardDim)
        pass


    def __moveEvent(self):
        k = 32
        pass

if __name__ == '__main__':
    
    start_app(ComputerVisionAlgorithm)