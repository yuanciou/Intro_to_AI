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
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        """
        #Minimax Search

        End of Recursion
        1. If the depth equals to the given depth equals to the given depth 'self.depth', or the state is the 
           winning state or losing state. The recursion should stop by returning the 
           'self.evaluationFunction(state)', which can compute the score of state.
        
        Each Recursion
        1. Create the list 'Value', which record the score of each state in each recursion.
        2. Use for loop to calculate the score of each legal action, the record the score in 'Value'.
        
        Return of The Recursion
        1. If it is the root the the decision tree(agent is Pacman and depth == 0), return the index of the 
           Pacman's legal actions which has the highest score.
        2. If the agent is Pacman, return the maximum value in 'Value'.
        3. If the agent is ghost, return the minimum value in 'Value'.
        """
        
        def minimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            Value = []

            for action in state.getLegalActions(agentIndex):
                nextState = state.getNextState(agentIndex, action)
                
                if agentIndex == gameState.getNumAgents() - 1:
                    Value.append(minimax(nextState, depth + 1, 0))
                else:
                    nextagent = agentIndex + 1
                    Value.append(minimax(nextState, depth, nextagent))

            if (agentIndex == 0 and depth == 0):
                maxValue = max(Value)
                idx = Value.index(maxValue)
                return idx

            if (agentIndex == 0):
                return max(Value)
            else:
                return min(Value)

        return gameState.getLegalActions(0)[minimax(gameState, 0, 0)]
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        """
        #Alpha-Beta Pruning

        End of Recursion
        1. If the depth equals to the given depth equals to the given depth 'self.depth', or the state is the 
           winning state or losing state. The recursion should stop by returning the 
           'self.evaluationFunction(state)', which can compute the score of state.

        Each Recursion
        1. If the current agent is Pacman
           a. Initial the 'value' to negative infinity
           b. Use a for loop to do run the recursion for each legal action.
           c. Use 'value' to rememeber the maximum score of 'alpha_beta_pruning()'.
           d. If the value is bigger than 'beta', return value.
           e. Update 'alpha' if 'Value' is larger than 'alpha'.
        2. If the current agent is ghost
           a. Initial the 'value' to positive infinity.
           b. Use a for loop to do run the recursion for each legal action.
           c. Use 'value' to rememeber the minimum score of 'alpha_beta_pruning()'.
           d. If the value is smaller than 'alpha', return value.
           e. Update 'beta' if 'Value' is samller than 'beta'. 

        #Outside the function

        Define the variables
        1. Use 'legal_actions' to record the legal action of initial Pacman.
        2. Initial 'best_action' to None for return the best action.
        3. Initial 'best_value' and 'alpha' to negative infinity, and 'beta' to 
           positive infinity.
        
        Recursion Part
        1. Use 'value' to record the socre return by 'alpha_beta_pruning'.
        2. If the score is larger than 'best_value', record the score and update
           'best_action' to the action of this score.
        3. Update 'alpha' if 'Value' is larger than 'alpha'.
        """
        def alpha_beta_pruning(state, depth, agentIndex, alpha, beta):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIndex == 0:
                value = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    next_state = state.getNextState(agentIndex, action)
                    value = max(value, alpha_beta_pruning(next_state, depth, agentIndex + 1, alpha, beta))
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                value = float("inf")
                for action in state.getLegalActions(agentIndex):
                    next_state = state.getNextState(agentIndex, action)
                    if agentIndex == gameState.getNumAgents() - 1:
                        value = min(value, alpha_beta_pruning(next_state, depth + 1, 0, alpha, beta))
                    else:
                        value = min(value, alpha_beta_pruning(next_state, depth, agentIndex + 1, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value

        legal_actions = gameState.getLegalActions(0)
        best_action = None
        best_value = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for action in legal_actions:
            next_state = gameState.getNextState(0, action)
            value = alpha_beta_pruning(next_state, 0, 1, alpha, beta)
            if value > best_value:
                best_value = value
                best_action = action
            alpha = max(alpha, best_value)

        return best_action
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        """
        #Expectimax Search

        End of Recursion
        1. If the depth equals to the given depth equals to the given depth 'self.depth', or the state is the 
           winning state or losing state. The recursion should stop by returning the 
           'self.evaluationFunction(state)', which can compute the score of state.
        
        Each Recursion
        1. Create the list 'Value', which record the score of each state in each recursion.
        2. Use for loop to calculate the score of each legal action, the record the score in 'Value'.
        
        Return of The Recursion
        1. If it is the root the the decision tree(agent is Pacman and depth == 0), return the index of the 
           Pacman's legal actions which has the highest score.
        2. If the agent is Pacman, return the maximum value in 'Value'.
        3. If the agent is ghost, return the mean value of 'Value' since the ghosts' action is random.
        """
        def expectimax(state, depth, agentIndex):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            
            Value = []

            for action in state.getLegalActions(agentIndex):
                nextState = state.getNextState(agentIndex, action)
                
                if agentIndex == gameState.getNumAgents() - 1:
                    Value.append(expectimax(nextState, depth + 1, 0))
                else:
                    nextagent = agentIndex + 1
                    Value.append(expectimax(nextState, depth, nextagent))

            if (agentIndex == 0 and depth == 0):
                maxValue = max(Value)
                idx = Value.index(maxValue)
                return idx

            if (agentIndex == 0):
                return max(Value)
            else:
                return sum(Value)/len(Value)

        return gameState.getLegalActions(0)[expectimax(gameState, 0, 0)]
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """
    Define Variables
    1. 'score' for current score, 'currentPosition' for current position, 'list_of_food' is
       a list record current foods, 'list_of_capsule' record current capsules, 'state_of_ghost'
       record current ghosts
    2. 'number_of_food' record the number of food, initial 'min_food_distance', 'min_capsule_distance',
       and 'min_scare_ghost' as negative infinity.

    Calculate Variables
    1. 'min_food_distance' would be the minimum distance of food.
    2. 'min_capsule_distance' would be the minimum distance of capsule.
    3. If the Pacman was eaten by the ghost the score should be negative infinity.
    4. 'min_scare_ghost' would be the minimum distance of scared ghost.

    Evaluate Score
    1. Consider current score.
    2. If the distance of food, capsule, or scared ghost is shorter, the score will be higher.
    3. If the number of food is more, the score should be lower.
    """
    score = currentGameState.getScore()
    currentPosition = currentGameState.getPacmanPosition()
    list_of_food = currentGameState.getFood().asList()
    list_of_capsule = currentGameState.getCapsules()
    state_of_ghost = currentGameState.getGhostStates()
    
    number_of_food = len(list_of_food)
    min_food_distance = float('inf')
    min_capsule_distance = float('inf')
    min_scare_ghost = float('inf')

    for food in list_of_food:
        min_food_distance = min(min_food_distance, manhattanDistance(currentPosition, food))
    for capsule in list_of_capsule:
        min_capsule_distance = min(min_capsule_distance, manhattanDistance(currentPosition, capsule))
    for ghost in state_of_ghost:
        if manhattanDistance(currentPosition, ghost.getPosition()) == 0 and ghost.scaredTimer == 0:
            return float("-inf")
        if ghost.scaredTimer > 0:
            min_scare_ghost = min(min_scare_ghost, manhattanDistance(currentPosition, ghost.getPosition()))
        
    food_score = 10 / min_food_distance
    capsule_score = 50 / min_capsule_distance
    ghost_score = 200 / min_scare_ghost
    
    return score + food_score + capsule_score + ghost_score - number_of_food
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
