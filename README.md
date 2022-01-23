# ChessVision
ChessVision uses Keras Sequential API to make a convolutional neural network for the purpose of determining the positions of chess pieces on a chessboard. ChessVisionModelTraining is the Python code to train the model on images of empty and occupied - black and occupied - white pieces so that an image of a full chessboard can be parsed into individual squares and the model can predict if each square is empty or has a white or black piece on it. 

ChessVision is the skeleton user interface where a user can set up a new chess game or training session with their webcam positioned overhead of the chessboard. The user is prompted to trace the playing area on the webcam feed and those dimensions are used to divide the board into 64 regions of interest. When the player makes a move, they hit the space bar, much as they would hit a a chess clock, and the program takes a snapshot of the board and parses each space to see which ones have changed status since the last turn. 

Uses the logic that all pieces have a particular starting square and with each move, one square will become unoccupied and another square will be either become occupied or change color if a piece is taken (except in cases of castling where both the king and a rook move places). Based on which turn of the game it is, the program checks if pieces of the color whose turn it was are still occuppying their squares - checking until a space is found where the piece is no longer occupying the space. It then looks for spaces that have changed occupancy status and concludes that the piece missing from its previous spot has moved to that space. 

Future Growth: 
- Relay moves to another website for play or broadcast --> alternative to expensive DGT boards
- Track sequences of moves to check accuracy against opening variations or endgame theory. Audio cue or visual cue from the program to indicate an error. 
- Stats tracking 
- Connect image collection to a real chess clock


Why care about training with a physical board? 
Success in chess often comes down to ability to memorize thousands of moves of openings prep, remember complex endgame patterns and recognize tactical patterns in game. There are many popular training websites like Chessable, Chess.com, Lichess.org where the typical presentation of the chessboard is vertical board. This presentation is much different from the horizontal view of over the board chess, training engrains patterns of a digital board and not a real board. Players must also develop good habits such as not touching a piece until they are ready to make a move (locked into moving that piece if you touch it) and getting in the habit of making game notations and pressing a chess clock. 
