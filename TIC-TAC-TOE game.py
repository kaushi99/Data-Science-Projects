X="x"
O="O"
NUM_SQUARES=9
TIE="TIE"
EMPTY=" "


def display_instructions():
    """Display game instructions."""


    print(

    """
    Welcome to the greatest intellectual challenge of all time: Tic-Tac-Toe.
    This will be a showdown between your human brain and my silicon processor.
    You will make your move known by entering a number, 0 - 8. The number
    will correspond to the board position as illustrated:
    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

    Prepare yourself, human. The ultimate battle is about to begin. \n
    """
    )
def ask_yes_no(question):
      response= None
      while response not in ("y","n"):
            response=raw_input(question).lower()
      return response    
            
def pieces():
      go_first=ask_yes_no("Do you require the first move? (y/n)\n")
      if go_first=="y":
            human=X
            computer=O
      else:
            computer=X
            human=O
      return computer,human

def new_board():
      board=[]
      for sq in range(NUM_SQUARES):
            board.append(EMPTY)
      return board

def display_board(board):
      """Display game board on screen."""
      print "\n\t", board[0], "|", board[1], "|", board[2]
      print "\t", "---------"
      print "\t", board[3], "|", board[4], "|", board[5]
      print "\t", "---------"
      print "\t", board[6], "|", board[7], "|", board[8], "\n"


def winner(board):
       WAYS_TO_WIN = ((0, 1, 2),
                      (3, 4, 5),
                      (6, 7, 8),
                      (0, 3, 6),
                      (1, 4, 7),
                      (2, 5, 8),
                      (0, 4, 8),
                      (2, 4, 6))

       for row in WAYS_TO_WIN:
             if board[row[0]]== board[row[1]]==board[row[2]] !=EMPTY:
                   winner=board[row[0]]
                   return winner
             if EMPTY not in board:
                   return TIE
             return None


def legal_move(board):
      moves=[ ]
      for sq in range(NUM_SQUARES):
            if board[sq]==EMPTY:
                  moves.append(sq)
                  
      return moves


def ask_number(question,low,high):
      response=None
      while response not in range(low,high):
            response=int(input(question))
      return response


def human_move(board,human):
      legal=legal_move(board)
      move = None
      while move not in legal:
            move=ask_number("Where will you move? (0-8)",0,NUM_SQUARES)
            if move not in legal:
                  print("Thatsquare is already ocupied,foolish human. Choose another.\n")
      print("Fine...")
      return move


def computer_move(board,computer,human):
            board=board[:]
            best_moves=(4,0,2,6,8,1,3,5,7)
            print("I shall take square number")
            for move in legal_move(board):
                  board[move]=computer
                  if winner(board)==computer:
                      print(move)
                      return move
                  board[move]=EMPTY

            
            for move in legal_move(board):
                 board[move]=human
                 if winner(board)==human:
                      print(move)
                      return move
                 board[move]=EMPTY
            for move in best_moves:
                  if move in legal_move(board):
                        print(move)
                        return move

def next_turn(turn):
            if turn==X:
                return O
            else:
                return X
            

def congrat_winner(the_winner, computer, human):

    """Congratulate the winner."""
    if the_winner != TIE:
        print(the_winner, "won!\n")
    else:
        print("It's a tie!\n")
    if the_winner == computer:
        print("As I predicted, human, I am triumphant once more. \n" \
        "Proof that computers are superior to humans in all regards.")
    elif the_winner == human:
        print("No, no! It cannot be! Somehow you tricked me, human. \n" \
        "But never again! I, the computer, so swear it!")
    elif the_winner == TIE:
        print("You were most lucky, human, and somehow managed to tie me. \n" \
        "Celebrate today... for this is the best you will ever achieve.")


            
def main():
      display_instructions()
      computer,human=pieces()
      turn=X
      board=new_board()
      display_board(board)

      while not winner(board):
            if turn == human:
                 move=human_move(board,human)
                 board[move]=human
            else:
                move=computer_move(board,computer,human)
                board[move]=computer

            display_board(board)
            turn=next_turn(turn)
      the_winner=winner(board)
      congrat_winner(the_winner,computer,human)     



main()
raw_input("PRESS ENTER TO EXIT:")
            
