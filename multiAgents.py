# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print("newScaredTimes",newScaredTimes)



        currentFood = currentGameState.getFood()
        #print("currentFood",len(currentFood.asList()))
        #print("newFood",len(newFood.asList()))
        dis_P_ghost=[]
        ghost_positions=[s.getPosition() for s in newGhostStates]
        for ghost in ghost_positions:
            dis_P_ghost.append(manhattanDistance(newPos,ghost))
            #print([ghostState.scaredTimer for ghostState in newGhostStates])
        min_distance_ghost=min(dis_P_ghost)
        if min_distance_ghost==0:
            return -1000

        distance_P_food=[]
        for food in currentFood.asList():
            distance_P_food.append(manhattanDistance(newPos,food))

        minimum_to_food=min(distance_P_food)
        return -minimum_to_food

        "*** YOUR CODE HERE ***"

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """

    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def min_value(self, gameState, depth, agentindex):
        #v = 99999
        min_scores=[]
        actions=[]

        for action in gameState.getLegalActions(agentindex):
            child = gameState.generateSuccessor(agentindex, action)

            if agentindex == gameState.getNumAgents()-1:
                min_scores.append(self.value(child, depth - 1, 0)[0])
            else:
                min_scores.append(self.value(child, depth, agentindex + 1)[0])

            actions.append(action)

        min_score=min(min_scores)
        min_index = [i for i, score in enumerate(min_scores) if score == min_score]
        doaction=actions[random.choice(min_index)]
        return min_score, doaction



    def max_value(self, gameState, depth, agentindex):
        #v = -99999
        max_scores=[]
        actions=[]

        for action in gameState.getLegalActions(agentindex):
            child = gameState.generateSuccessor(agentindex, action)
            if agentindex == gameState.getNumAgents() - 1:
                max_scores.append(self.max_value(child, depth - 1, 0)[0])
            else:
                max_scores.append(self.value(child, depth, agentindex + 1)[0])
            actions.append(action)

        max_score = max(max_scores)
        max_index = [i for i, score in enumerate(max_scores) if score == max_score]
        doaction = actions[random.choice(max_index)]
        return max_score,doaction




    def value(self, gameState, depth, agentindex):
        if depth ==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""

        if agentindex==0:
            #print("firsttttt step")
            return self.max_value(gameState,depth,agentindex)
        else:
            #print("second step")
            return self.min_value(gameState, depth, agentindex)

    def getAction(self, gameState):

        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
          Returns the total number of agents in the game

          gameState.isWin():
          Returns whether or not the game state is a winning state

          gameState.isLose():
          Returns whether or not the game state is a losing state
          """

        #print("########################### New RUN ##########################")
        action = self.value(gameState, self.depth, 0)[1]
        return action


        ##use a stack to do the recursion of the utilities
        ###calculate utilities for the leave nodes first and then if max using the max between
        # the most bottom otherwise the min


        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def min_value(self, gameState, depth, agentindex,alpha,beta):
        #v = 99999
        min_scores=[]
        actions=[]

        for action in gameState.getLegalActions(agentindex):
            child = gameState.generateSuccessor(agentindex, action)

            if agentindex == gameState.getNumAgents()-1:
                min_scores.append(self.value(child, depth - 1, 0,alpha,beta)[0])
            else:
                min_scores.append(self.value(child, depth, agentindex + 1,alpha,beta)[0])

            actions.append(action)
            min_score=min(min_scores)
            if min_score < alpha:
                return min_score, ""
            beta = min(beta, min_score)

        min_index = [i for i, score in enumerate(min_scores) if score == min_score]
        doaction = actions[random.choice(min_index)]

        return min_score, doaction




    def max_value(self, gameState, depth, agentindex,alpha,beta):
        #v = -99999
        max_scores=[]
        actions=[]

        for action in gameState.getLegalActions(agentindex):
            child = gameState.generateSuccessor(agentindex, action)
            if agentindex == gameState.getNumAgents() - 1:
                max_scores.append(self.max_value(child, depth - 1, 0,alpha,beta)[0])
            else:
                max_scores.append(self.value(child, depth, agentindex + 1,alpha,beta)[0])
            actions.append(action)

            max_score = max(max_scores)
            if max_score > beta:
                return max_score, ""
            alpha = max(alpha, max_score)

        max_index = [i for i, score in enumerate(max_scores) if score == max_score]
        doaction = actions[random.choice(max_index)]
        return max_score,doaction




    def value(self, gameState, depth, agentindex,alpha,beta):
        if depth ==0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""

        if agentindex==0:
            return self.max_value(gameState,depth,agentindex,alpha,beta)
        else:
            return self.min_value(gameState, depth, agentindex,alpha,beta)

    def getAction(self, gameState):

        """
         Returns the minimax action using self.depth and self.evaluationFunction
         """
        #print("########################### New RUN ##########################")
        action = self.value(gameState, self.depth, 0,-999999,999999)[1]
        return action



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expect_value(self, gameState, depth, agentindex):
        # v = 99999
        min_scores = []
        actions = []
        legalMoves=gameState.getLegalActions(agentindex)
        probability=1/len(legalMoves)
        for action in legalMoves:
            child = gameState.generateSuccessor(agentindex, action)

            if agentindex == gameState.getNumAgents() - 1:
                min_scores.append(self.value(child, depth - 1, 0)[0])
            else:

                min_scores.append(self.value(child, depth, agentindex + 1)[0]*probability)

        min_score = sum(min_scores)
        doaction = legalMoves[len(legalMoves)-1]
        return min_score, doaction

    def max_value(self, gameState, depth, agentindex):
        # v = -99999
        max_scores = []
        actions = []

        for action in gameState.getLegalActions(agentindex):
            child = gameState.generateSuccessor(agentindex, action)
            if agentindex == gameState.getNumAgents() - 1:
                max_scores.append(self.max_value(child, depth - 1, 0)[0])
            else:
                max_scores.append(self.value(child, depth, agentindex + 1)[0])
            actions.append(action)

        max_score = max(max_scores)
        max_index = [i for i, score in enumerate(max_scores) if score == max_score]
        doaction = actions[random.choice(max_index)]
        return max_score, doaction

    def value(self, gameState, depth, agentindex):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""

        if agentindex == 0:
            # print("firsttttt step")
            return self.max_value(gameState, depth, agentindex)
        else:
            # print("second step")
            return self.expect_value(gameState, depth, agentindex)

    def getAction(self, gameState):

        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.

        """

        "*** YOUR CODE HERE ***"
        #print("########################### New RUN ##########################")
        action = self.value(gameState, self.depth, 0)[1]
        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"


    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    currentPosition = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()
    #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    currentFood = currentGameState.getFood()
    scaredTimes =[]
    dis_P_ghost = []
    dis_P_capsule=[]
    distance_P_food = []
    dis_capsule_ghost=[]
    ghost_positions = [s.getPosition() for s in newGhostStates]


    """
    #nearest capsule
    capsules = currentGameState.getCapsules()
    for capsule in capsules:
        dis_P_capsule.append(manhattanDistance(currentPosition, capsule))
    if len(dis_P_capsule)!=0:
        min_distance_capsule=min(dis_P_capsule)

    ####distance capsule gost
    
    for capsule in capsules:
        for ghost in ghost_positions:
            dis_capsule_ghost.append(manhattanDistance(capsule, ghost))

    min_ghost_capsule=min(dis_capsule_ghost)
    """
    
    ######
    #nearestgost and scared time

    for ghost in ghost_positions:
        dis_P_ghost.append(manhattanDistance(currentPosition, ghost))
        scaredTimes=[ghostState.scaredTimer for ghostState in newGhostStates]
    scaredTime=scaredTimes[0]
    min_distance_ghost = min(dis_P_ghost)
    if scaredTime>0 and min_distance_ghost<=1:
        return 10000
    else:
        if min_distance_ghost == 0:
            return -10000



    for food in currentFood.asList():
        distance_P_food.append(manhattanDistance(currentPosition, food))
    if len(distance_P_food)==0:
        minimum_to_food = 0
    else:
        minimum_to_food = min(distance_P_food)
    if scaredTime > 0:
        value = currentGameState.getScore() - minimum_to_food - min_distance_ghost
    else:
        value = currentGameState.getScore() - minimum_to_food + min_distance_ghost
    return value


# Abbreviation
better = betterEvaluationFunction
