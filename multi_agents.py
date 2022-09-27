"""
multiAgents.py
--------------
Licensing Information:  You are free to use or extend these projects for
educational purposes provided that (1) you do not distribute or publish
solutions, (2) you retain this notice, and (3) you provide clear
attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

Attribution Information: The Pacman AI projects were developed at UC Berkeley.
The core projects and autograders were primarily created by John DeNero
(denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
"""


import random
import util

from util import manhattan_distance
from game import Agent, Directions


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)  # Pick randomly among the best

        # Add more of your code here if you want to

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghost_state.scared_timer for ghost_state in new_ghost_states]
        # print('successor_game_state.get_score()')
        # print(successor_game_state.get_score())
        # print('new_pos')
        # print(new_pos)
        # print('new_food')
        # print(new_food.as_list())
        # print('new_ghost_states')
        # print(new_ghost_states.get_ghost_state)
        # print('new_scared_times')
        # print(new_scared_times)

        # *** YOUR CODE HERE ***
        # return successor_game_state.get_score()
        score = 0
        if successor_game_state.is_win():
            return 999999

        # check ghost positions
        # manhattan_distance(xy1, xy2):
        # new_food.asList()
        foodLocations = new_food.as_list()
        ghostStates = []
        distanceFromGhostNextState = []
        distanceFromGhostCurrentState = []

        # need distance to ghosts
        # this is the next state location of ghosties
        for ghostLoc in new_ghost_states:
            # print(ghostLoc.get_position())
            distance = manhattan_distance(new_pos, ghostLoc.get_position())
            distanceFromGhostNextState.append(distance)

        #current state location of ghosties
        for ghostLoc in current_game_state.get_ghost_states():
            distance = manhattan_distance(new_pos, ghostLoc.get_position())
            distanceFromGhostCurrentState.append(distance)

        # need distance to food
        distanceFromFood = []
        for foodLoc in foodLocations:
            distance = manhattan_distance(new_pos, foodLoc)
            # print('distance')
            # print(distance)
            distanceFromFood.append(distance)

        # How to set up score? Just random numbers I come up with?


        # Probably need to do something with pellets
        # Need to add/remove from score based on these things

        # if ghosts are scared need to make score better when closer to the ghosties

        # if ghosts are not scared make score worse when close

        # if food nearby score better

        # if food far away score worse

        # if pellet ? score better (probably same as food)





        return score


def score_evaluation_function(current_game_state):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return current_game_state.get_score()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action from the current game_state using self.depth
          and self.evaluation_function.

          Here are some method calls that might be useful when implementing minimax.

          game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means Pacman, ghosts are >= 1

          game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action

          game_state.get_num_agents():
            Returns the total number of agents in the game
        """

        # *** YOUR CODE HERE ***

        util.raise_not_defined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluation_function
        """

        # *** YOUR CODE HERE ***

        util.raise_not_defined()


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
          Returns the expectimax action using self.depth and self.evaluation_function

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        # *** YOUR CODE HERE ***

        util.raise_not_defined()


def better_evaluation_function(current_game_state):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    # *** YOUR CODE HERE ***

    # Useful information you can extract from a GameState (pacman.py)
    successor_game_state = current_game_state
    new_pos = successor_game_state.get_pacman_position()
    new_food = successor_game_state.get_food()
    new_ghost_states = successor_game_state.get_ghost_states()

    util.raise_not_defined()


# Abbreviation
better = better_evaluation_function
