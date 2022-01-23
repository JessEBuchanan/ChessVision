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
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.models import model_from_json
import os


class chessSquare:
     
    def __init__(self, name, startRow, endRow, startCol, endCol): #name is the square name, space is the subarray range within the chessboard
        
        self.name = name
        self.StartRow = startRow #may not need to adjust these numbers
        self.EndRow = endRow
        self.StartCol = startCol
        self.EndCol = endCol
        self.occupied = 0

class chessPiece: 
    def __init__(self, name, start):
        self.name = name
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

    empty = []
    for i in range(2,6):
        for j in range(0,8):
            empty.append(chessPiece('empty', spaces[i][j]))
 
    
    blackPieces = []
    whitePieces = []

    blackPieces.append(chessPiece('blackLightRook',spaces[7][7]))
    blackPieces.append(chessPiece('blackDarkKnight', spaces[7][6]))
    blackPieces.append(chessPiece('blackLightBishop', spaces[7][5]))
    blackPieces.append(chessPiece('blackQueen', spaces[7][4]))
    blackPieces.append(chessPiece('blackKing', spaces[7][3]))
    blackPieces.append(chessPiece('blackDarkBishop', spaces[7][2]))
    blackPieces.append(chessPiece('blackLightKnight', spaces[7][1]))
    blackPieces.append(chessPiece('blackDarkRook', spaces[7][0]))
    blackPieces.append(chessPiece('blackAPawn', spaces[6][7]))
    blackPieces.append(chessPiece('blackBPawn', spaces[6][6]))
    blackPieces.append(chessPiece('blackCPawn', spaces[6][5]))
    blackPieces.append(chessPiece('blackDPawn', spaces[6][4]))
    blackPieces.append(chessPiece('blackEPawn', spaces[6][3]))
    blackPieces.append(chessPiece('blackFPawn', spaces[6][2]))
    blackPieces.append(chessPiece('blackGPawn', spaces[6][1]))
    blackPieces.append(chessPiece('blackHPawn', spaces[6][0]))

    whitePieces.append(chessPiece('whiteDarkRook', spaces[0][7]))
    whitePieces.append(chessPiece('whiteLightKnight', spaces[0][6]))
    whitePieces.append(chessPiece('whiteDarkBishop', spaces[0][5]))
    whitePieces.append(chessPiece('whiteQueen', spaces[0][4]))
    whitePieces.append(chessPiece('whiteKing', spaces[0][3]))
    whitePieces.append(chessPiece('whiteLightBishop',  spaces[0][2]))
    whitePieces.append(chessPiece('whiteDarkKnight',  spaces[0][1]))
    whitePieces.append(chessPiece('whiteLightRook',  spaces[0][0]))
    whitePieces.append(chessPiece('whiteAPawn',  spaces[1][7]))
    whitePieces.append(chessPiece('whiteBPawn',  spaces[1][6]))
    whitePieces.append(chessPiece('whiteCPawn',  spaces[1][5]))
    whitePieces.append(chessPiece('whiteDPawn',  spaces[1][4]))
    whitePieces.append(chessPiece('whiteEPawn', spaces[1][3]))
    whitePieces.append(chessPiece('whiteFPawn',  spaces[1][2]))
    whitePieces.append(chessPiece('whiteGPawn',  spaces[1][1]))
    whitePieces.append(chessPiece('whiteHPawn',  spaces[1][0]))

    return spaces, blackPieces, whitePieces, empty

def checkOccupied(chessSpace, image):
    space = image[chessSpace.StartRow:chessSpace.EndRow, chessSpace.StartCol:chessSpace.EndCol]
    imageResize = cv2.resize(space, (60,60), cv2.INTER_LINEAR)
    grayArray = np.zeros((1,60,60,3))
    grayArray[0] = imageResize
    y_pred = loaded_model.predict(grayArray)
    #print(y_pred)
    pred = np.argmax(y_pred[0])
    return pred

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


def newBoard():
    newBoard = staticROI()
    board = newBoard.croppedPlay
    boardDim = getBoardDim(board)
    return board, boardDim
    

def checkPosition(frame,turn, whitePieces, blackPieces, empty): 
    
    if turn%2 == 0: #check if black's turn or white
        print('Black Turn: ', turn)
        movedPiece = blackPieces[len(blackPieces)-1]
        for piece in blackPieces:
            space = checkOccupied(piece.current, frame)
            if space ==0:
                movedPiece = piece
                break
        for piece in reversed(empty):
            space = checkOccupied(piece.current, frame)
            if space != 0:
                filledSpace = piece
                blackPieces.remove(movedPiece)
                blackPieces.append(filledSpace)
                empty.remove(filledSpace)
                empty.append(movedPiece)
                print('Moved: ', movedPiece.name, 'from ', movedPiece.current.name, 'to ', filledSpace.current.name)
                movedPiece.name, filledSpace.name = filledSpace.name, movedPiece.name #swap spaces
                return
        for piece in whitePieces:
                space = checkOccupied(piece.current, frame)
                if space !=2: #indicates taken piece
                    movedPiece[0].current = piece.current
                    whitePieces.remove(piece) # remove piece from the game
                    blackPieces.remove(movedPiece)
                    blackPieces.append(piece)
                    print(movedPiece.name, 'took ', piece.name, 'on ', filledSpace.current.name)
                    movedPiece.name, piece.name = piece.name, movedPiece.name #swap spaces
                    return

    else: #white's turn
        print('White Turn: ', turn)
        movedPiece = whitePieces[len(whitePieces)-1] # ensures this has a value if moved piece not found
        for piece in whitePieces:
            space = checkOccupied(piece.current, frame)
            if space ==0:
                movedPiece = piece
                break
        for piece in empty:
            space = checkOccupied(piece.current, frame)
            if space != 0:
                filledSpace = piece
                whitePieces.remove(movedPiece)
                whitePieces.append(filledSpace)
                empty.remove(filledSpace)
                empty.append(movedPiece)
                print('Moved: ', movedPiece.name, 'from ', movedPiece.current.name, 'to ', filledSpace.current.name)
                movedPiece.name, filledSpace.name = filledSpace.name, movedPiece.name #swap spaces
                return
        for piece in blackPieces:
                space = checkOccupied(piece.current, frame)
                if space !=1: #indicates taken piece
                    movedPiece[0].current = piece.current
                    blackPieces.remove(piece) # remove piece from the game
                    whitePieces.remove(movedPiece)
                    whitePieces.append(piece)
                    print(movedPiece.name, 'took ', piece.name, 'on ', filledSpace.current.name)
                    movedPiece.name, piece.name = piece.name, movedPiece.name #swap spaces
                    return
    print('Move not determined')


board = cv2.imread('C:/Users/Jess/Documents/ChessVision/Training/Advanced Variation/start.jpg')
boardDim = getBoardDim(board)
boardSpaces = np.zeros((8,8)) 
blackPieces = []
whitePieces = []
empty = []
k = 0
json_file = open('C:/Users/Jess/Documents/ChessVision/Training/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("C:/Users/Jess/Documents/ChessVision/Training/model.h5")
loaded_model.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])
loaded_model.summary()

def main():
    input('Press enter when ready to start a game\n')
    turn = 1
    print('Wait for the webcam to appear, then press c. A duplicate of the webcam will appear, drag and hold the mouse button to draw a rectangle around just the playing square.')
    print('If you do not like the square, right click to redraw. When done, press c and then press r.')
    print('When you make your move, press the space bar to trigger recording the move')
    print('When game is done, press q to quit')
    board, boardDim = newBoard()
    boardSpaces, blackPieces, whitePieces, empty = newGame(board, boardDim)
    #check starting positions
    for i in range (0,8):
        for j in range(0,8):
            print(checkOccupied(boardSpaces[i][j], board))

    videoCaptureObject = cv2.VideoCapture(0)
    while(True):
        ret, frame = videoCaptureObject.read()
        cv2.imshow('Capturing Play',frame)
        k = cv2.waitKey(1)
        if(k == 32):
            cv2.imwrite('C:/Users/Jess/Documents/ChessVision/Training/Webcam/newpic.jpg',frame) 
            checkPosition(frame, turn, whitePieces, blackPieces, empty)
            turn = turn + 1
        if(cv2.waitKey(1) == ord('q')):
            videoCaptureObject.release()
            cv2.destroyAllWindows()
            exit()

if __name__ == "__main__":
    main()


    
    
    
    
