import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob
import math
import PIL

class chessSquare:
     
    def __init__(self, name, startRow, endRow, startCol, endCol, startImage): #name is the square name, space is the subarray range within the chessboard
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
        self.StartRow = startRow + 5
        self.EndRow = endRow - 5
        self.StartCol = startCol +32
        self.EndCol = endCol
        self.color = squares[name]
        self.occupied = False

class chessPiece: 
    def __init__(self, name,color, start):
        self.name = name
        self.color = color
        self.current = start
        
def setUp(image): #determines approximate dimensions of each square
    boardDim = image.shape[0]
    boardDim = boardDim//8
    return boardDim


def checkStart(spaces, image): #checks that the starting spaces in ranks 1-2 and 7-8 are occupied
    for i in range (0,2):
        for j in range(0,8):
            occupied = checkOccupied(image,spaces[i][j],)
            if occupied == False: 
                return False

    for i in range (6,8):
        for j in range(0,8):
            occupied = checkOccupied(image,spaces[i][j])
            if occupied == False: 
                return False
    return True



def newGame(board, dim): #uses dimensions of the board image to partition into the individual spaces. Create chesspiece objects and set their starting current position. 
   spaces = np.zeros((8,8), dtype=chessSquare)   
   files = ['h', 'g', 'f', 'e', 'd','c','b','a']
   ranks = ['1','2','3','4','5','6','7','8']

   for i in range(0,8):
       for j in range(0,8):
           spaces[i][j]=chessSquare(files[j]+ranks[i],(dim*i),(dim*(i+1)),(dim*j),(dim*(j+1)),board)
 
   pieces = []
   pieces.append(chessPiece('blackLightRook','black',spaces[7][7].name))
   pieces.append(chessPiece('blackDarkKnight', 'black', spaces[7][6].name))
   pieces.append(chessPiece('blackLightBishop', 'black', spaces[7][5].name))
   pieces.append(chessPiece('blackQueen', 'black', spaces[7][4].name))
   pieces.append(chessPiece('blackKing', 'black', spaces[7][3].name))
   pieces.append(chessPiece('blackDarkBishop', 'black', spaces[7][2].name))
   pieces.append(chessPiece('blackLightKnight', 'black', spaces[7][1].name))
   pieces.append(chessPiece('blackDarkRook', 'black', spaces[7][0].name))
   pieces.append(chessPiece('blackAPawn', 'black', spaces[6][7].name))
   pieces.append(chessPiece('blackBPawn', 'black', spaces[6][6].name))
   pieces.append(chessPiece('blackCPawn', 'black', spaces[6][5].name))
   pieces.append(chessPiece('blackDPawn', 'black', spaces[6][4].name))
   pieces.append(chessPiece('blackEPawn', 'black', spaces[6][3].name))
   pieces.append(chessPiece('blackFPawn', 'black', spaces[6][2].name))
   pieces.append(chessPiece('blackGPawn', 'black', spaces[6][1].name))
   pieces.append(chessPiece('blackHPawn', 'black', spaces[6][0].name))

   pieces.append(chessPiece('whiteDarkRook','white', spaces[0][7].name))
   pieces.append(chessPiece('whiteLightKnight', 'white', spaces[0][6].name))
   pieces.append(chessPiece('whiteDarkBishop', 'white', spaces[0][5].name))
   pieces.append(chessPiece('whiteQueen', 'white', spaces[0][4].name))
   pieces.append(chessPiece('whiteKing', 'white', spaces[0][3].name))
   pieces.append(chessPiece('whiteLightBishop', 'white', spaces[0][2].name))
   pieces.append(chessPiece('whiteDarkKnight', 'white', spaces[0][1].name))
   pieces.append(chessPiece('whiteLightRook', 'white', spaces[0][0].name))
   pieces.append(chessPiece('whiteAPawn', 'white', spaces[1][7].name))
   pieces.append(chessPiece('whiteBPawn', 'white', spaces[1][6].name))
   pieces.append(chessPiece('whiteCPawn', 'white', spaces[1][5].name))
   pieces.append(chessPiece('whiteDPawn', 'white', spaces[1][4].name))
   pieces.append(chessPiece('whiteEPawn','white', spaces[1][3].name))
   pieces.append(chessPiece('whiteFPawn', 'white', spaces[1][2].name))
   pieces.append(chessPiece('whiteGPawn', 'white', spaces[1][1].name))
   pieces.append(chessPiece('whiteHPawn', 'white', spaces[1][0].name))

   return spaces,pieces

#def main():

    #board = cv2.imread('C:/Users/Jess/Documents/ChessVision/blankBoardCropped.jpg')
    #boardDim = setUp(board)
    #boardSpaces, pieces = newGame(board, boardDim)    
    #updateImage = cv2.imread('C:/Users/Jess/Documents/ChessVision/startBoard.jpg')
    

    #img = board[boardSpaces[0][0].StartRow:boardSpaces[0][0].EndRow, boardSpaces[0][0].StartCol: boardSpaces[0][0].EndCol]
    #cv2.imshow("Original", img)
    #cv2.waitKey(0)
    #upImg = updateImage[boardSpaces[0][0].StartRow:boardSpaces[0][0].EndRow, boardSpaces[0][0].StartCol: boardSpaces[0][0].EndCol]
    #cv2.imshow("Update", upImg)
    #cv2.waitKey(0)
    #print('done')
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    
