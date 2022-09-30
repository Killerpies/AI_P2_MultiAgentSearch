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
        # *** YOUR CODE HERE ***
        foodLocationsNextState = new_food.as_list()
        distanceFromFoodNextState = []
        distanceFromGhostNextState = []
        distanceFromFoodCurrentState = []
        score = successor_game_state.get_score()
        # Next Game State
        for foodLoc in foodLocationsNextState:
            distanceFromFoodNextState.append(manhattan_distance(foodLoc, new_pos))
        for ghostLoc in new_ghost_states:
            distanceFromGhostNextState.append(manhattan_distance(ghostLoc.get_position(), new_pos))
        #Current Game State
        for foodLoc in current_game_state.get_food().as_list():
            distanceFromFoodCurrentState.append(manhattan_distance(foodLoc, new_pos))
        if successor_game_state.is_win():
            return 999999999
        #Ghosty to close
        if min(distanceFromGhostNextState) <= 2:
            return -999999999
        score += 2 * min(distanceFromGhostNextState)
        score -= 2 * min(distanceFromFoodNextState)

        #if closer to food
        if min(distanceFromFoodCurrentState) < min(distanceFromFoodNextState):
            # 10 * max distance of food
            score += 10 * max(distanceFromFoodNextState)

        if action == Directions.STOP:
            score -= 10
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
        # print(game_state.get_legal_actions(0))
        # print(game_state.get_legal_actions(1))
        # game_state.generate_successor(0, "Center")
        # game_state.generate_successor(0, "Center")
        # game_state.generate_successor(1, "Center")
        # game_state.generate_successor(1, "Center")
        #
        # print(game_state.get_num_agents())
        # print(self.depth)
        # print(self.evaluation_function(game_state.generate_successor(0, "Center")))
        return self.max_value(game_state, 0)

    def max_value(self, game_state, depth):
        if game_state.is_win() or game_state.is_lose():
            return game_state.get_score()
        actions = game_state.get_legal_actions(0)
        # max func starts with -inf
        tempscore = highestScore = float("-inf")
        best_action = Directions.STOP
        for action in actions:
            # score = min function
            tempscore = self.min_value_recursive(game_state.generate_successor(0, action), depth, 1)
            # get max score here
            if tempscore > highestScore:
                highestScore = tempscore
                best_action = action
        # if depth is 0 then we found the best action
        if depth == 0:
            return best_action
        else:
            # otherwise return the highscore back to min function
            return highestScore

    def min_value_recursive(self, game_state, depth, agent):
        if game_state.is_lose() or game_state.is_win():
            return game_state.get_score()
        # increment agent index + 1
        nextAgent = agent + 1
        # if at the end of agent list, reset back to pacman
        if agent == game_state.get_num_agents() - 1:
            nextAgent = 0
        actions = game_state.get_legal_actions(agent)
        # min func starts with +inf
        tempscore = highestScore = float("inf")
        for action in actions:
            if nextAgent == 0:
                # if next agent is packman and we r at proper depth
                # set tempscore to the evaluationfunction
                if depth == self.depth - 1:
                    tempscore = self.evaluation_function(game_state.generate_successor(agent, action))
                else:
                    # if not pacman or proper depth, tempscore does another run of max incrementing depth
                    tempscore = self.max_value(game_state.generate_successor(agent, action), depth + 1)
            else:
                # if not pacman then tempscore runs this function again with the next ghost/agent
                tempscore = self.min_value_recursive(game_state.generate_successor(agent, action), depth, nextAgent)
            # return min value here
            if tempscore < highestScore:
                highestScore = tempscore
        # return min
        return highestScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
          Returns the minimax action using self.depth and self.evaluation_function
        """

        # *** YOUR CODE HERE ***
        alpha = float("-inf")
        beta = float("+inf")

        return self.max_value(game_state, 0, alpha, beta)

    def max_value(self, game_state, depth, alpha, beta):
        if game_state.is_win() or game_state.is_lose():
            return game_state.get_score()
        actions = game_state.get_legal_actions(0)
        # max func starts with -inf
        tempscore = highestScore = float("-inf")
        best_action = Directions.STOP
        for action in actions:
            # score = min function
            tempscore = self.min_value_recursive(game_state.generate_successor(0, action), depth, 1)
            # get max score here
            if tempscore > highestScore:
                highestScore = tempscore
                best_action = action
        # if depth is 0 then we found the best action
        if depth == 0:
            return best_action
        else:
            # otherwise return the highscore back to min function
            return highestScore

    def min_value_recursive(self, game_state, depth, agent, alpha, beta):
        if game_state.is_lose() or game_state.is_win():
            return game_state.get_score()
        # increment agent index + 1
        nextAgent = agent + 1
        # if at the end of agent list, reset back to pacman
        if agent == game_state.get_num_agents() - 1:
            nextAgent = 0
        actions = game_state.get_legal_actions(agent)
        # min func starts with +inf
        tempscore = highestScore = float("inf")
        for action in actions:
            if nextAgent == 0:
                # if next agent is packman and we r at proper depth
                # set tempscore to the evaluationfunction
                if depth == self.depth - 1:
                    tempscore = self.evaluation_function(game_state.generate_successor(agent, action))
                else:
                    # if not pacman or proper depth, tempscore does another run of max incrementing depth
                    tempscore = self.max_value(game_state.generate_successor(agent, action), depth + 1, alpha, beta)
            else:
                # if not pacman then tempscore runs this function again with the next ghost/agent
                tempscore = self.min_value_recursive(game_state.generate_successor(agent, action), depth, nextAgent, alpha, beta)
            # return min value here
            if tempscore < highestScore:
                highestScore = tempscore
        # return min
        return highestScore

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
