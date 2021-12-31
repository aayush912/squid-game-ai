import time
from BaseAI import BaseAI
from Grid import Grid
from Utils import *

# TO BE IMPLEMENTED
#

time_limit = 4
early_finish = 0.2


class PlayerAI(BaseAI):

    def __init__(self) -> None:
        # You may choose to add attributes to your player - up to you!
        super().__init__()
        self.pos = None
        self.player_num = None
        self.max_level = 4

    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def move_heuristic(self, grid):


        curr_pos = grid.find(self.player_num)
        neighbors = grid.get_neighbors(curr_pos, only_available=True)

        edge_dist = sorted([curr_pos[0], 6 - curr_pos[0], curr_pos[1], 6 - curr_pos[1]])

        # Want to be away from the corners and edges
        min_edge_dist = 2 * edge_dist[0] + edge_dist[1]

        player_moves = len(neighbors)
        for n in neighbors:
            player_moves += len(grid.get_neighbors(n, only_available=True))

        neighbors = grid.get_neighbors(grid.find(3 - self.player_num), only_available=True)

        opp_moves = len(neighbors)
        for n in neighbors:
            opp_moves += len(grid.get_neighbors(n, only_available=True))

        return 1.5 * player_moves - 0.5*opp_moves + 2 * min_edge_dist

    def trap_heuristic(self, grid):
        neighbors = grid.get_neighbors(grid.find(self.player_num), only_available=True)

        player_moves = len(neighbors)
        for n in neighbors:
            player_moves += len(grid.get_neighbors(n, only_available=True))

        neighbors = grid.get_neighbors(grid.find(3 - self.player_num), only_available=True)

        opp_moves = len(neighbors)
        for n in neighbors:
            opp_moves += len(grid.get_neighbors(n, only_available=True))

        return player_moves - 1.5 * opp_moves

    # check if over time
    def time_constraint_failed(self, currTime):
        if currTime - self.start_time > time_limit - early_finish:
            return True
        else:
            return False

    # includes both the minimize and expectation steps; and alpha beta pruning
    def move_minimize(self, grid, level, alpha, beta):

        if level == 0 or self.time_constraint_failed(time.time()):
            return grid.find(self.player_num), self.move_heuristic(grid)

        curr_pos = grid.find(self.player_num)

        opp_player_pos = grid.find(3 - self.player_num)
        neighbors = grid.get_neighbors(curr_pos, only_available=True)

        child = neighbors[0] if neighbors and len(neighbors) > 0 else curr_pos
        worst_utility = np.inf

        for neighbor in neighbors:
            g2 = grid.clone()
            g2.trap(neighbor)

            probability = 1 - 0.05 * (manhattan_distance(opp_player_pos, neighbor) - 1)
            _, utility = self.move_maximize(g2, level, alpha, beta)

            utility *= probability

            if utility < worst_utility:
                child, worst_utility = neighbor, utility
            if worst_utility <= alpha:
                break
            if worst_utility < beta:
                beta = worst_utility
        return child, worst_utility

    # maximization step
    def move_maximize(self, grid, level, alpha, beta):

        if level == 0 or self.time_constraint_failed(time.time()):
            return grid.find(self.player_num), self.move_heuristic(grid)

        max_utility = -np.inf

        curr_pos = grid.find(self.player_num)
        neighbors = grid.get_neighbors(curr_pos, only_available=True)

        child = neighbors[0] if neighbors and len(neighbors) > 0 else curr_pos

        for neighbor in neighbors:
            g2 = grid.clone()
            g2.move(neighbor, self.player_num)

            _, utility = self.move_minimize(g2, level - 1, alpha, beta)
            if utility > max_utility:
                child, max_utility = neighbor, utility
            if max_utility >= beta:
                break
            if max_utility > alpha:
                alpha = max_utility
        return child, max_utility

    def getMove(self, grid):
        self.start_time = time.time()

        decision, _ = self.move_maximize(grid, self.max_level, -np.inf, np.inf)

        print("Time taken for move: ", time.time() - self.start_time)
        return decision

    # includes both the minimize and expectation steps; and alpha beta pruning
    def trap_minimize(self, grid, level, alpha, beta):

        if level == 0 or self.time_constraint_failed(time.time()):
            return grid.find(3 - self.player_num), self.trap_heuristic(grid)

        opp_player_pos = grid.find(3 - self.player_num)
        neighbors = grid.get_neighbors(opp_player_pos, only_available=True)

        child = neighbors[0] if neighbors and len(neighbors) > 0 else opp_player_pos
        worst_utility = np.inf

        for neighbor in neighbors:
            g2 = grid.clone()
            g2.move(neighbor, 3-self.player_num)

            _, utility = self.trap_maximize(g2, level, alpha, beta)

            if utility < worst_utility:
                child, worst_utility = neighbor, utility
            if worst_utility <= alpha:
                break
            if worst_utility < beta:
                beta = worst_utility
        return child, worst_utility

    # maximization step
    def trap_maximize(self, grid, level, alpha, beta):

        if level == 0 or self.time_constraint_failed(time.time()):
            return grid.find(3 - self.player_num), self.move_heuristic(grid)

        maxUtility = -np.inf

        curr_pos = grid.find(self.player_num)

        opp_player_pos = grid.find(3 - self.player_num)
        neighbors = grid.get_neighbors(opp_player_pos, only_available=True)

        child = neighbors[0] if neighbors and len(neighbors) > 0 else opp_player_pos

        for neighbor in neighbors:
            g2 = grid.clone()
            g2.trap(neighbor)

            probability = 1 - 0.05 * (manhattan_distance(curr_pos, neighbor) - 1)
            _, utility = self.trap_minimize(g2, level - 1, alpha, beta)

            utility *= probability

            if utility > maxUtility:
                child, maxUtility = neighbor, utility
            if maxUtility >= beta:
                break
            if maxUtility > alpha:
                alpha = maxUtility
        return child, maxUtility


    def getTrap(self, grid: Grid):
        self.start_time = time.time()

        decision, _ = self.trap_maximize(grid, self.max_level, -np.inf, np.inf)

        print("Time taken for move: ", time.time() - self.start_time)
        return decision
