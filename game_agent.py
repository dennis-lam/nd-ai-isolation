"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import math

NO_LEGAL_MOVES = -1, -1

is_initialized = False
game_height = 0
game_width = 0
longest_distance = 0
corner_locations = []

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def initialize_global_variables(game):
    """ Instead of re-create instance every time,
        initialize global variables one time in order to improve the performance.
    """
    global is_initialized, game_width, game_height, longest_distance, corner_locations

    if is_initialized:
        return

    # Assign information.
    game_height = game.height
    game_width = game.width
    longest_distance = game_width - 1 + game_height - 1
    corner_locations = [(0, 0), (0, game_width - 1), (game_height - 1, 0), (game_height - 1, game_width - 1)]
    is_initialized = True

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Initialize global variables if needed.
    initialize_global_variables(game)

    # Initialize local variables
    is_own_active = game.active_player == player
    opp = game.get_opponent(player)
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opp)
    own_moves_count = len(own_moves)
    opp_moves_count = len(opp_moves)
    own_location = game.get_player_location(player)
    opp_location = game.get_player_location(opp)
    blank_spaces = game.get_blank_spaces()

    # 1. Implementation of "Overlap Move"
    # Must avoid this.
    if own_moves_count == 1 and own_moves[0] in opp_moves:
        return float("-inf")
    # Favor
    if opp_moves_count == 1 and opp_moves[0] in own_moves:
        return float("inf")
    # Opposite move may block future move.
    if is_own_active:
        overlap_moves = [opp_move for opp_move in opp_moves if opp_move in own_moves]
        if len(overlap_moves) > 0:
            opp_moves_count = opp_moves_count - 1
    else:
        overlap_moves = [own_move for own_move in own_moves if own_move in opp_moves]
        if len(overlap_moves) > 0:
            own_moves_count = own_moves_count - 1

    # 2. Implementation of "Corner Move"
    # Adjustment of own move
    corner_moves = [own_move for own_move in own_moves if own_move in corner_locations]
    if len(corner_moves) > 0:
        own_moves_count = own_moves_count - 0.8 * len(corner_moves)
    # Adjustment of opposite move
    corner_moves = [opp_move for opp_move in opp_moves if opp_move in corner_locations]
    if len(corner_moves) > 0:
        opp_moves_count = opp_moves_count - 0.8 * len(corner_moves)

    # 3. Obtain diagonal chain factor
    diagonal_factor = get_diagonal_factor(own_location, blank_spaces)
    # 4. Obtain distance factor
    distance_factor = get_distance_factor(own_location, opp_location)
    # Find net value.
    net_count = own_moves_count - opp_moves_count
    # Calculate the score
    return net_count + net_count * diagonal_factor * 0.1 + net_count * (1- distance_factor) * 0.1


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Initialize global variables if needed.
    initialize_global_variables(game)

    # Initialize local variables
    is_own_active = game.active_player == player
    opp = game.get_opponent(player)
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opp)
    own_moves_count = len(own_moves)
    opp_moves_count = len(opp_moves)
    own_location = game.get_player_location(player)
    opp_location = game.get_player_location(opp)
    blank_spaces = game.get_blank_spaces()

    # 1. Implementation of "Overlap Move"
    # Must avoid this.
    if own_moves_count == 1 and own_moves[0] in opp_moves:
        return float("-inf")
    # Favor
    if opp_moves_count == 1 and opp_moves[0] in own_moves:
        return float("inf")
    # Opposite move may block future move.
    if is_own_active:
        overlap_moves = [opp_move for opp_move in opp_moves if opp_move in own_moves]
        if len(overlap_moves) > 0:
            opp_moves_count = opp_moves_count - 1
    else:
        overlap_moves = [own_move for own_move in own_moves if own_move in opp_moves]
        if len(overlap_moves) > 0:
            own_moves_count = own_moves_count - 1

    # 2. Implementation of "Corner Move"
    # Adjustment of own move
    corner_moves = [own_move for own_move in own_moves if own_move in corner_locations]
    if len(corner_moves) > 0:
        own_moves_count = own_moves_count - 0.8 * len(corner_moves)
    # Adjustment of opposite move
    corner_moves = [opp_move for opp_move in opp_moves if opp_move in corner_locations]
    if len(corner_moves) > 0:
        opp_moves_count = opp_moves_count - 0.8 * len(corner_moves)

    # 3. Obtain diagonal chain factor
    diagonal_factor = get_diagonal_factor(own_location, blank_spaces)
    # 4. Obtain distance factor
    distance_factor = get_distance_factor(own_location, opp_location)
    # Calculate the score
    return own_moves_count - opp_moves_count + own_moves_count * diagonal_factor * 0.25 + own_moves_count * (1- distance_factor) * 0.25


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # Initialize global variables if needed.
    initialize_global_variables(game)

    # Initialize local variables
    is_own_active = game.active_player == player
    opp = game.get_opponent(player)
    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opp)
    own_moves_count = len(own_moves)
    opp_moves_count = len(opp_moves)

    # 1. Implementation of "Overlap Move"
    # Must avoid this.
    if own_moves_count == 1 and own_moves[0] in opp_moves:
        return float("-inf")
    # Favor
    if opp_moves_count == 1 and opp_moves[0] in own_moves:
        return float("inf")
    # Opposite move may block future move.
    if is_own_active:
        overlap_moves = [opp_move for opp_move in opp_moves if opp_move in own_moves]
        if len(overlap_moves) > 0:
            opp_moves_count = opp_moves_count - 1
    else:
        overlap_moves = [own_move for own_move in own_moves if own_move in opp_moves]
        if len(overlap_moves) > 0:
            own_moves_count = own_moves_count - 1

    # 2. Implementation of "Corner Move"
    # Adjustment of own move
    corner_moves = [own_move for own_move in own_moves if own_move in corner_locations]
    if len(corner_moves) > 0:
        own_moves_count = own_moves_count - 0.8 * len(corner_moves)
    # Adjustment of opposite move
    corner_moves = [opp_move for opp_move in opp_moves if opp_move in corner_locations]
    if len(corner_moves) > 0:
        opp_moves_count = opp_moves_count - 0.8 * len(corner_moves)

    return float(own_moves_count - opp_moves_count)


def get_diagonal_factor(location, blank_spaces):
    """ Check top left, top right, bottom left and bottom right locations are filled.
        :returns range 0 ~ 1. 1 means all 4 locations are filled. 0 means no locations are filled.
    """
    diagonal_locations = get_diagonal_locations(location)
    blank_locations = [location for location in diagonal_locations if location in blank_spaces]
    # Filled locations
    filled_diagonal_locations_count = len(diagonal_locations) - len(blank_locations)
    return filled_diagonal_locations_count * 0.25


def get_diagonal_locations(location):
    """  Obtain top left, top right, bottom left and bottom right locations.
    """
    locations = [(location[0] - 1, location[1] - 1), (location[0] - 1, location[1] + 1),
                 (location[0] + 1, location[1] - 1), (location[0] + 1, location[1] + 1)]
    return [(location_y, location_x) for location_y, location_x in locations
            if location_y > -1 and location_y < game_height and
            location_x > -1 and location_x < game_width]


def get_distance_factor(location1, location2):
    """ Distance rate between two location

        :returns range 0 ~ 1. The smaller the value, the shorter the distance.
    """
    distance = abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])
    return (distance - 1) / (longest_distance - 1)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = NO_LEGAL_MOVES
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            print('Search Timeout! Time Left : {} , Timer threshold : {}'.format(
                  self.time_left(), self.TIMER_THRESHOLD ))

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # Check timeout
        self.validate_timeout()
        # Obtain move-score dictionary
        move_score_dict = {move: self.get_min_score(game.forecast_move(move), depth-1)
                           for move in game.get_legal_moves()}
        # Returns the move with the highest scores if it contains more than 1 move
        # and returns (-1,-1) when no legal moves available
        if len(move_score_dict) > 0:
            return max(move_score_dict, key=lambda i: move_score_dict[i])
        else:
            return NO_LEGAL_MOVES

    def validate_timeout(self):
        """ Raise error when timeout.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def get_max_score(self, game, depth):
        """Obtain maximum score.

        Returns
        -------
        score

        """
        # Check timeout
        self.validate_timeout()
        # Obtain legal moves
        legal_moves = game.get_legal_moves()
        # Return information if needed
        if depth < 1 or len(legal_moves) < 1:
            return self.score(game, self)
        # Obtain maximum value from score list
        return max([self.get_min_score(game.forecast_move(move), depth-1) for move in legal_moves])

    def get_min_score(self, game, depth):
        """Obtain minimum score.

        Returns
        -------
        score

        """
        # Check timeout
        self.validate_timeout()
        # Obtain legal moves
        legal_moves = game.get_legal_moves()
        # Return information if needed
        if depth < 1 or len(legal_moves) < 1:
            return self.score(game, self)
        # Obtain minimum value from score list
        return min([self.get_max_score(game.forecast_move(move),depth-1) for move in legal_moves])


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.`
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = NO_LEGAL_MOVES
        depth = 1
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            # for depth in range(1, self.search_depth+1):
            #     best_move = self.alphabeta(game, depth)
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1

        except SearchTimeout:
            pass
            # print('Search Timeout in depth {}! Time Left : {} , Timer threshold : {}'.format(
            #     depth, self.time_left(), self.TIMER_THRESHOLD ))

        # Return the best move from the last completed search iteration
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # Check timeout
        self.validate_timeout()
        # Obtains and returns the best move
        max_move, _ = self.get_max_info(game, depth, alpha, beta)
        return max_move

    def validate_timeout(self):
        """ Raise error when timeout.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def get_max_info(self, game, depth, alpha, beta, last_move=NO_LEGAL_MOVES):
        """ Obtain maximum score and related information.

        Returns
        -------
        move of maximum score, maximum score
        """
        # Check timeout
        self.validate_timeout()
        # Obtain legal moves
        legal_moves = game.get_legal_moves()
        # Return information if needed
        if depth < 1 or len(legal_moves) < 1:
            return last_move, self.score(game, self)
        # Initialize maximum score and move information.
        max_move = NO_LEGAL_MOVES
        max_score = float("-inf")
        # Handle each legal move
        for move in legal_moves:
            # Obtain the score of next possible move
            _, score = self.get_min_info(game.forecast_move(move), depth - 1, alpha, beta, move)
            # Set current move as the best move if needed.
            if max_move == NO_LEGAL_MOVES or score > max_score:
                max_move = move
                max_score = score
            # Prune - No need to check remaining moves
            if score >= beta:
                break
            # Update alpha value if needed.
            if score > alpha:
                alpha = score
        # Returns maximum score and related information.
        return max_move, max_score

    def get_min_info(self, game, depth, alpha, beta, last_move=NO_LEGAL_MOVES):
        """ Obtain minimum score and related information.

        Returns
        -------
        move of minimum score, minimum score
        """
        # Check timeout
        self.validate_timeout()
        # Obtain legal moves
        legal_moves = game.get_legal_moves()
        # Return information if needed
        if depth < 1 or len(legal_moves) < 1:
            return last_move, self.score(game, self)
        # Initialize minimum score and move information.
        min_move = NO_LEGAL_MOVES
        min_score = float("inf")
        # Handle each legal move
        for move in legal_moves:
            # Obtain the score of next possible move
            _, score = self.get_max_info(game.forecast_move(move), depth - 1, alpha, beta, move)
            # Set current move as the best move if needed.
            if min_move == NO_LEGAL_MOVES or min_score > score:
                min_move = move
                min_score = score
            # Prune - No need to check remaining moves
            if score <= alpha:
                break
            # Update beta value if needed.
            if beta > score:
                beta = score
        # Returns minimum score and related information.
        return min_move, min_score
