# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util
from util import heappush, heappop, Stack, Queue, PriorityQueue
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
      """
      Returns the start state for the search problem
      """
      util.raiseNotDefined()

    def isGoalState(self, state):
      """
      state: Search state

      Returns True if and only if the state is a valid goal state
      """
      util.raiseNotDefined()

    def getSuccessors(self, state):
      """
      state: Search state

      For a given state, this should return a list of triples,
      (successor, action, stepCost), where 'successor' is a
      successor to the current state, 'action' is the action
      required to get there, and 'stepCost' is the incremental
      cost of expanding to that successor
      """
      util.raiseNotDefined()

    def getCostOfActions(self, actions):
      """
      actions: A list of actions to take

      This method returns the total cost of a particular sequence of actions.  The sequence must
      be composed of legal moves
      """
      util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure that you implement the graph search version of DFS,
    which avoids expanding any already visited states.
    Otherwise your implementation may run infinitely!
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    """
    YOUR CODE HERE
    """
    actions = []
    closedset = set()
    openset = Stack()
    startNode = (None, None, problem.getStartState())
    openset.push(startNode)
    while not openset.isEmpty():
        node = openset.pop()
        parentNode,action,state=node
        if problem.isGoalState(state):
            while state != problem.getStartState():
                actions.append(action)
                (parentNode,action,state) = parentNode
            actions.reverse()
            return actions

        if state not in closedset:
            closedset.add(state)
            for (successor, next_action, stepCost) in problem.getSuccessors(state):
                childNode = (node, next_action, successor)
                openset.push(childNode)
    return actions


def breadthFirstSearch(problem):
    """
    YOUR CODE HERE
    """
    actions = []
    closedset = set()
    openset = Queue()
    startNode = (None, None, problem.getStartState())
    openset.push(startNode)
    while not openset.isEmpty():
        node = openset.pop()
        parentNode,action,state=node
        if problem.isGoalState(state):
            while state != problem.getStartState():
                actions.append(action)
                (parentNode,action,state) = parentNode
            actions.reverse()
            return actions

        if state not in closedset:
            closedset.add(state)
            for (successor, next_action, stepCost) in problem.getSuccessors(state):
                childNode = (node, next_action, successor)
                openset.push(childNode)
    return actions
def uniformCostSearch(problem):
    """
    YOUR CODE HERE
    """
    actions = []
    closedset = set()
    openset = PriorityQueue()
    startNode = (None, None, problem.getStartState(),0)
    openset.push(startNode,0)
    while not openset.isEmpty():
        node = openset.pop()
        parentNode,action,state,cost=node
        if problem.isGoalState(state):
            while state != problem.getStartState():
                actions.append(action)
                (parentNode,action,state,priority) = parentNode
            actions.reverse()
            return actions

        if state not in closedset:
            closedset.add(state)
            for (successor, next_action, stepCost) in problem.getSuccessors(state):
                priority=cost+stepCost
                childNode = (node, next_action, successor,priority)
                openset.push(childNode,priority)
    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """
    YOUR CODE HERE
    """
    actions = []
    closedset = set()
    openset = PriorityQueue()
    h0 = heuristic(problem.getStartState(), problem)
    startNode = (None, None, problem.getStartState(),h0)
    openset.push(startNode,h0)
    while not openset.isEmpty():
        node = openset.pop()
        parentNode,action,state,cost=node
        if problem.isGoalState(state):
            while state != problem.getStartState():
                actions.append(action)
                (parentNode,action,state,priority) = parentNode
            actions.reverse()
            return actions

        if state not in closedset:
            closedset.add(state)
            for (successor, next_action, stepCost) in problem.getSuccessors(state):
                gn=cost+stepCost
                childNode = (node, next_action, successor,gn)
                openset.push(childNode,gn+heuristic(successor,problem))
    return actions


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
