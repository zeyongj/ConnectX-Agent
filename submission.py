# submission.py
def agent(observation, configuration):
    import numpy as np
    import random

    # Constants
    ROWS = configuration.rows
    COLUMNS = configuration.columns
    IN_A_ROW = configuration.inarow

    # Piece definitions
    EMPTY = 0
    PLAYER_PIECE = observation.mark
    OPPONENT_PIECE = 3 - observation.mark  # Assuming player marks are 1 or 2

    def drop_piece(board, row, col, piece):
        board[row][col] = piece

    def is_valid_location(board, col):
        return board[0][col] == EMPTY

    def get_valid_locations(board):
        valid_locations = []
        for col in range(COLUMNS):
            if is_valid_location(board, col):
                valid_locations.append(col)
        return valid_locations

    def get_next_open_row(board, col):
        for r in range(ROWS-1, -1, -1):
            if board[r][col] == EMPTY:
                return r

    def winning_move(board, piece):
        # Check horizontal locations for win
        for c in range(COLUMNS - IN_A_ROW + 1):
            for r in range(ROWS):
                if all([board[r][c + i] == piece for i in range(IN_A_ROW)]):
                    return True

        # Check vertical locations for win
        for c in range(COLUMNS):
            for r in range(ROWS - IN_A_ROW + 1):
                if all([board[r + i][c] == piece for i in range(IN_A_ROW)]):
                    return True

        # Check positively sloped diagonals
        for c in range(COLUMNS - IN_A_ROW + 1):
            for r in range(ROWS - IN_A_ROW + 1):
                if all([board[r + i][c + i] == piece for i in range(IN_A_ROW)]):
                    return True

        # Check negatively sloped diagonals
        for c in range(COLUMNS - IN_A_ROW + 1):
            for r in range(IN_A_ROW - 1, ROWS):
                if all([board[r - i][c + i] == piece for i in range(IN_A_ROW)]):
                    return True
        return False

    def evaluate_window(window, piece):
        score = 0
        opp_piece = OPPONENT_PIECE

        if window.count(piece) == IN_A_ROW:
            score += 100
        elif window.count(piece) == IN_A_ROW - 1 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(piece) == IN_A_ROW - 2 and window.count(EMPTY) == 2:
            score += 2

        if window.count(opp_piece) == IN_A_ROW - 1 and window.count(EMPTY) == 1:
            score -= 4

        return score

    def score_position(board, piece):
        score = 0

        ## Score center column
        center_array = [int(i) for i in list(board[:, COLUMNS // 2])]
        center_count = center_array.count(piece)
        score += center_count * 3

        ## Score Horizontal
        for r in range(ROWS):
            row_array = [int(i) for i in list(board[r, :])]
            for c in range(COLUMNS - IN_A_ROW + 1):
                window = row_array[c:c + IN_A_ROW]
                score += evaluate_window(window, piece)

        ## Score Vertical
        for c in range(COLUMNS):
            col_array = [int(i) for i in list(board[:, c])]
            for r in range(ROWS - IN_A_ROW + 1):
                window = col_array[r:r + IN_A_ROW]
                score += evaluate_window(window, piece)

        ## Score positive sloped diagonal
        for r in range(ROWS - IN_A_ROW + 1):
            for c in range(COLUMNS - IN_A_ROW + 1):
                window = [board[r + i][c + i] for i in range(IN_A_ROW)]
                score += evaluate_window(window, piece)

        ## Score negative sloped diagonal
        for r in range(IN_A_ROW - 1, ROWS):
            for c in range(COLUMNS - IN_A_ROW + 1):
                window = [board[r - i][c + i] for i in range(IN_A_ROW)]
                score += evaluate_window(window, piece)

        return score

    def is_terminal_node(board):
        return winning_move(board, PLAYER_PIECE) or winning_move(board, OPPONENT_PIECE) or len(get_valid_locations(board)) == 0

    def minimax(board, depth, alpha, beta, maximizingPlayer):
        valid_locations = get_valid_locations(board)
        is_terminal = is_terminal_node(board)
        if depth == 0 or is_terminal:
            if is_terminal:
                if winning_move(board, PLAYER_PIECE):
                    return (None, 100000000000000)
                elif winning_move(board, OPPONENT_PIECE):
                    return (None, -100000000000000)
                else:  # Game is over, no more valid moves
                    return (None, 0)
            else:  # Depth is zero
                return (None, score_position(board, PLAYER_PIECE))
        if maximizingPlayer:
            value = -np.Inf
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                row = get_next_open_row(board, col)
                temp_board = board.copy()
                drop_piece(temp_board, row, col, PLAYER_PIECE)
                new_score = minimax(temp_board, depth - 1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    best_col = col
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_col, value
        else:  # Minimizing player
            value = np.Inf
            best_col = random.choice(valid_locations)
            for col in valid_locations:
                row = get_next_open_row(board, col)
                temp_board = board.copy()
                drop_piece(temp_board, row, col, OPPONENT_PIECE)
                new_score = minimax(temp_board, depth - 1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    best_col = col
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return best_col, value

    # Convert the observation board to a 2D board
    board = np.array(observation.board).reshape(ROWS, COLUMNS)
    # Decide which move to make
    valid_locations = get_valid_locations(board)
    if len(valid_locations) == 0:
        return random.randint(0, COLUMNS - 1)
    else:
        # Use Minimax to choose the best column
        col, minimax_score = minimax(board, 4, -np.Inf, np.Inf, True)
        if col is not None and is_valid_location(board, col):
            return col
        else:
            # If Minimax fails, pick a random valid column
            return random.choice(valid_locations)
