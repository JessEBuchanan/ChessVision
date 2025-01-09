#!/usr/bin/env python
# coding: utf-8

# # ChessVision - Pandas Structure for Chess Board

# In[2]:


import numpy as np
import pandas as pd


# In[51]:


def newGameBoard():
    board = pd.DataFrame(index = [1,2,3,4,5,6,7,8])
    board.loc[1, 'A'] = 'White Rook'
    board.loc[1, 'B'] = 'White Knight'
    board.loc[1, 'C'] = 'White Bishop'
    board.loc[1, 'D'] = 'White Queen'
    board.loc[1, 'E'] = 'White King'
    board.loc[1, 'F'] = 'White Bishop'
    board.loc[1, 'G'] = 'White Knight'
    board.loc[1, 'H'] = 'White Rook'

    board.loc[8, 'A'] = 'Black Rook'
    board.loc[8, 'B'] = 'Black Knight'
    board.loc[8, 'C'] = 'Black Bishop'
    board.loc[8, 'D'] = 'Black Queen'
    board.loc[8, 'E'] = 'Black King'
    board.loc[8, 'F'] = 'Black Bishop'
    board.loc[8, 'G'] = 'Black Knight'
    board.loc[8, 'H'] = 'Black Rook'

    file = 'A'
    for i in range(1,9):
        board.loc[2, file] = 'White Pawn'
        board.loc[7, file] = 'Black Pawn'
        board.loc[3,file] = "Unoccupied"
        board.loc[4,file] = "Unoccupied"
        board.loc[5,file] = "Unoccupied"
        board.loc[6,file] = "Unoccupied"
        file = chr(ord(file) + 1)
    return board
def newLocationBoard():
    board = pd.DataFrame(index = [1,2,3,4,5,6,7,8])
    file = 'A'
    for i in range(1,9):
        board.loc[1, file] = 'Occupied White'
        board.loc[2, file] = 'Occupied White'
        board.loc[7, file] = 'Occupied Black'
        board.loc[8, file] = 'Occupied Black'
        board.loc[3,file] = "Unoccupied"
        board.loc[4,file] = "Unoccupied"
        board.loc[5,file] = "Unoccupied"
        board.loc[6,file] = "Unoccupied"
        file = chr(ord(file) + 1)
    return board
        
gameBoard = newGameBoard()
board1 = newLocationBoard()
board2 = newLocationBoard()
print(board1)
board2.loc[4, 'F'] = 'Occupied'
board2.loc[1,'G'] = 'Unoccupied'
print(board2)


# In[62]:


def changeRowAndFile(diff):
    if diff.loc[diff.index[0],diff.columns[0][0]]['other'] == 'Unoccupied':
        toMove = diff.columns[2][0]+str(diff.index[1])
        fromMove = diff.columns[0][0]+str(diff.index[0])
        if diff.loc[diff.index[0],diff.columns[2][0]]['other'] == 'Unoccupied':
            fromMove, toMove = determineEnPassant(diff)
    elif diff.loc[diff.index[0],diff.columns[2][0]]['other'] == 'Unoccupied':
        toMove = diff.columns[0][0]+str(diff.index[1])
        fromMove = diff.columns[2][0]+str(diff.index[0])
        if diff.loc[diff.index[0],diff.columns[0][0]]['other'] == 'Unoccupied':
            fromMove, toMove = determineEnPassant(diff)
    elif diff.loc[diff.index[1],diff.columns[0][0]]['other'] == 'Unoccupied':
        toMove = diff.columns[2][0]+str(diff.index[0])
        fromMove = diff.columns[0][0]+str(diff.index[1])
        if diff.loc[diff.index[1],diff.columns[2][0]]['other'] == 'Unoccupied':
            fromMove, toMove = determineEnPassant(diff)
    elif diff.loc[diff.index[1],diff.columns[2][0]]['other'] == 'Unoccupied': 
        toMove = diff.columns[0][0]+str(diff.index[0])
        fromMove = diff.columns[2][0]+str(diff.index[1])
        if diff.loc[diff.index[1],diff.columns[0][0]]['other'] == 'Unoccupied':
            fromMove, toMove = determineEnPassant(diff)
    
    return fromMove, toMove
def determineEnPassant(diff):
    
    if pd.isna(diff.loc[diff.index[0],diff.columns[0][0]]['other']):
        toMove = diff.columns[2][0]+str(diff.index[0]) +' takes ' + diff.columns[2][0]+str(diff.index[1]) + ' with en passant'
        fromMove = diff.columns[0][0]+str(diff.index[1]) 
    elif pd.isna(diff.loc[diff.index[0],diff.columns[2][0]]['other']):
        
        toMove = diff.columns[0][0]+str(diff.index[0]) +' takes ' + diff.columns[0][0]+str(diff.index[1]) + ' with en passant'
        fromMove = diff.columns[2][0]+str(diff.index[1])
    elif pd.isna(diff.loc[diff.index[1],diff.columns[0][0]]['other']):
        
        toMove = diff.columns[2][0]+str(diff.index[1]) +' takes ' + diff.columns[2][0]+str(diff.index[0]) + ' with en passant'
        fromMove = diff.columns[0][0]+str(diff.index[0])
    elif pd.isna(diff.loc[diff.index[1],diff.columns[2][0]]['other']):
        
        toMove = diff.columns[0][0]+str(diff.index[1]) +' takes ' + diff.columns[0][0]+str(diff.index[0]) + ' with en passant'
        fromMove = diff.columns[2][0]+str(diff.index[0])
    
    return fromMove, toMove
                                         
def changeSameFile(diff):
    if diff.loc[diff.index[0],diff.columns[0][0]]['other'] == 'Unoccupied':
        toMove = diff.columns[0][0]+str(diff.index[1])
        fromMove = diff.columns[0][0]+str(diff.index[0])
        
    elif diff.loc[diff.index[1],diff.columns[0][0]]['other'] == 'Unoccupied':
        toMove = diff.columns[0][0]+str(diff.index[0])
        fromMove = diff.columns[0][0]+str(diff.index[1])
    
    return fromMove, toMove
    
def changeSameRow(diff):
    if diff.loc[diff.index[0],diff.columns[0][0]]['other'] == 'Unoccupied':
        toMove = diff.columns[2][0]+str(diff.index[0])
        fromMove = diff.columns[0][0]+str(diff.index[0])
    elif diff.loc[diff.index[0],diff.columns[2][0]]['other'] == 'Unoccupied':
        toMove = diff.columns[0][0]+str(diff.index[0])
        fromMove = diff.columns[2][0]+str(diff.index[0])
    
    return fromMove, toMove
  
def detectGameMove(board1, board2):
    diff = board1.compare(board2)
    move = 'Undefined move'
    if diff.shape == (2,4):
        fromMove, toMove = changeRowAndFile(diff)
        move = fromMove + ' to ' + toMove
    elif diff.shape == (2,2):
        fromMove, toMove = changeSameFile(diff)
        move = fromMove + ' to ' + toMove
    elif diff.shape == (1,4):
        fromMove, toMove = changeSameRow(diff)
        move = fromMove + ' to ' + toMove
    elif diff.shape == (1,8):
        if diff.index[0] == 1:
            move = 'White King castles short 0-0'
        elif diff.index[0]==8:
            move = 'Black King castles short 0-0'
    elif diff.shape == (1,10):
        if diff.index[0] == 1:
            move = 'White King castles long 0-0-0'
        elif diff.index[0]==8:
            move = 'Black King castles long 0-0-0'   
        
    return move
    
    
        
    #print(fromMove + ' to ' + toMove)
    #rowNames = diff.index
    #columnNames = diff.columns
    #print(rowNames[1])
    #print(columnNames[0][0])
    #print(diff.loc[2,'E'][0])
    #board2.loc[rowNames[1],columnNames[0][0]] = diff.loc[rowNames[0], 'self']
    #print(board2)
    #fromMove = diff.
#def updateGameBoard(fromMove, toMove, gameBoard):
    
    
move = detectGameMove(board1, board2)
print(move)
board3=newLocationBoard()
board3.loc[4, 'E'] = 'Occupied White'
board3.loc[1,'E'] = 'Unoccupied'
move = detectGameMove(board1, board3)
print(move)


# In[55]:


# Test simple move - G4 to E4

board4=newLocationBoard()
board4.loc[4, 'E'] = 'Occupied White'
board4.loc[4, 'G'] = 'Occupied Black'
board5=newLocationBoard()
board5.loc[4, 'E'] = 'Occupied Black'
board5.loc[4, 'G'] = 'Unoccupied'
move = detectGameMove(board4, board5)
print(move)


# In[56]:


# Test castling - black short castle. 

board6=newLocationBoard()
board7=newLocationBoard()

board6.loc[8,'F'] = 'Unoccupied'
board6.loc[8,'G'] = 'Unoccupied'

board7.loc[8,'E'] = 'Unoccupied'
board7.loc[8,'H'] = 'Unoccupied'

move = detectGameMove(board6, board7)
print(move)


# In[57]:


# Test en passant move - E5 to D6, taking D5 with en passant. 

board8=newLocationBoard()
board9=newLocationBoard()
board8.loc[7,'D'] = 'Unoccupied'
board8.loc[5,'D'] = 'Occupied Black'
board8.loc[2,'E'] = 'Unoccupied'
board8.loc[5,'E'] = 'Occupied White'
board9.loc[7,'D'] = 'Unoccupied'
board9.loc[2,'E'] = 'Unoccupied'
board9.loc[6,'D'] = 'Occupied White'
board9.loc[5,'E'] = 'Unoccupied'
board9.loc[5,'D'] = 'Unoccupied'
move = detectGameMove(board8, board9)
print(move)


# In[58]:


#Test en passant move - D4 to C3, taking C4 with en passant. 

board10=newLocationBoard()
board11=newLocationBoard()
board10.loc[7,'D'] = 'Unoccupied'
board10.loc[4,'D'] = 'Occupied Black'
board10.loc[2,'C'] = 'Unoccupied'
board10.loc[4,'C'] = 'Occupied White'
board11.loc[7,'D'] = 'Unoccupied'
board11.loc[2,'C'] = 'Unoccupied'
board11.loc[3,'C'] = 'Occupied Black'
board11.loc[4,'C'] = 'Unoccupied'
board11.loc[5,'D'] = 'Unoccupied'
move = detectGameMove(board10, board11)
print(move)


# In[63]:


# Test castling - black long castle. 

board6=newLocationBoard()
board7=newLocationBoard()

board6.loc[8,'B'] = 'Unoccupied'
board6.loc[8,'C'] = 'Unoccupied'
board6.loc[8,'D'] = 'Unoccupied'

board7.loc[8,'A'] = 'Unoccupied'
board7.loc[8,'E'] = 'Unoccupied'

move = detectGameMove(board6, board7)
print(move)


# In[64]:


# Test castling - white long castle. 

board6=newLocationBoard()
board7=newLocationBoard()

board6.loc[1,'B'] = 'Unoccupied'
board6.loc[1,'C'] = 'Unoccupied'
board6.loc[1,'D'] = 'Unoccupied'

board7.loc[1,'A'] = 'Unoccupied'
board7.loc[1,'E'] = 'Unoccupied'

move = detectGameMove(board6, board7)
print(move)


# In[65]:


# Test castling - white short castle. 

board6=newLocationBoard()
board7=newLocationBoard()

board6.loc[1,'F'] = 'Unoccupied'
board6.loc[1,'G'] = 'Unoccupied'

board7.loc[1,'E'] = 'Unoccupied'
board7.loc[1,'H'] = 'Unoccupied'

move = detectGameMove(board6, board7)
print(move)


# In[ ]:




