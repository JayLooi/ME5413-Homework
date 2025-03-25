from dataclasses import dataclass, field
from typing import Union
import heapq
import numpy as np
import math
import cv2


@dataclass(order=True)
class _Node:
    pos: Union[list, tuple] = field(compare=False)
    cost_so_far: float = field(compare=False)
    cost_f: float
    idx: int = field(compare=False)
    parent_idx: int = field(compare=False)


class _NodeQueue(list):
    def __init__(self):
        self.cost_f_list = dict()
        super().__init__()

    def __contains__(self, key):
        return key in self.cost_f_list

    def push(self, node):
        heapq.heappush(self, node)
        self.cost_f_list[node.idx] = node.cost_f

    def pop(self):
        node = heapq.heappop(self)

        if node.idx in self.cost_f_list:
            del self.cost_f_list[node.idx]

        return node

    def update(self, node):
        for i, item in enumerate(self):
            if item.idx == node.idx:
                self[i] = node
                break

        heapq.heapify(self)
        self.cost_f_list[node.idx] = node.cost_f


class GraphBasedPlanner:
    def __init__(self, workspace_map, map_resolution, agent_radius, search_method='a*',
                 heuristic_func=None):
        self.agent_radius = agent_radius
        self.map_resolution = map_resolution
        self._map_height, self._map_width = workspace_map.shape
        self._create_cspace_map(workspace_map)
        self._init_search_method(search_method, heuristic_func)

        # Actions with associated costs
        self._actions = (
            (0, -1, map_resolution),                        # Up
            (0, 1, map_resolution),                         # Down
            (-1, 0, map_resolution),                        # Left
            (1, 0, map_resolution),                         # Right
            (-1, -1, map_resolution * math.hypot(1, 1)),    # Upper-left
            (1, -1, map_resolution * math.hypot(1, 1)),     # Upper-right
            (-1, 1, map_resolution * math.hypot(1, 1)),     # Bottom-left
            (1, 1, map_resolution * math.hypot(1, 1)),      # Bottom-right
        )

    def _init_search_method(self, search_method, heuristic_func=None):
        self.search_method = search_method.lower()

        # Sanity check for algorithms that need a heuristic function
        if self.search_method in ('a*', 'gbfs'):
            if not callable(heuristic_func):
                raise TypeError((f'Argument heuristic_func for {search_method} '
                                f'must be callable but got {type(heuristic_func)}'))

            self._heuristic_func = heuristic_func

        if self.search_method == 'a*':
            self._create_node = lambda pos, cost, parent_idx: _Node(
                pos=pos,
                cost_so_far=cost,
                cost_f=cost + self._heuristic_func(pos, self.goal),
                idx=self._get_flatten_index(pos[1], pos[0]),
                parent_idx=parent_idx)

        elif self.search_method == 'dijkstra':
            self._create_node = lambda pos, cost, parent_idx: _Node(
                pos=pos,
                cost_so_far=cost,
                cost_f=cost,
                idx=self._get_flatten_index(pos[1], pos[0]),
                parent_idx=parent_idx)

        elif self.search_method == 'gbfs':   # greedy best-first search
            self._create_node = lambda pos, cost, parent_idx: _Node(
                pos=pos,
                cost_so_far=cost,
                cost_f=self._heuristic_func(pos, self.goal),
                idx=self._get_flatten_index(pos[1], pos[0]),
                parent_idx=parent_idx)

        else:
            raise ValueError(f'Invalid search_method {search_method}')

    def _create_cspace_map(self, workspace_map):
        # discretise agent diameter according to the workspace_map resolution
        diameter_pix = math.ceil(2 * self.agent_radius / self.map_resolution)

        # erode the boundary of foreground object (free space) by the size of agent_radius
        # which will produce the configuration space
        kernel = np.ones((diameter_pix + 1, diameter_pix + 1), dtype=np.uint8)
        self._cspace_map = cv2.erode(workspace_map, kernel, iterations=1)

    def _get_flatten_index(self, row, col):
        return row * self._map_width + col

    def _get_path(self, visited, goal_idx):
        idx = goal_idx
        path = []
        distance = visited[goal_idx].cost_so_far
        while idx is not None:
            node = visited[idx]
            path.append(node.pos)
            idx = node.parent_idx

        return path[::-1], distance, list(visited.keys())

    def plan(self, start, goal):
        self.start = start
        self.goal = goal
        visited = dict()
        queue = _NodeQueue()

        start_node = self._create_node(start, 0., None)
        queue.push(start_node)

        while queue:
            curr_node = queue.pop()
            curr_node_idx = curr_node.idx

            if curr_node_idx in visited:
                continue

            visited[curr_node_idx] = curr_node

            if tuple(curr_node.pos) == tuple(goal):
                return self._get_path(visited, curr_node_idx)

            for action in self._actions:
                col = curr_node.pos[0] + action[0]
                row = curr_node.pos[1] + action[1]

                # Check validity of action so that the agent will still be within map bound
                if (col < 0 or col >= self._map_width or
                    row < 0 or row >= self._map_height):
                    continue

                # Check if the action will result in obstacle collision
                if self._cspace_map[row][col] == 0:
                    continue

                cost = curr_node.cost_so_far + action[2]
                neighbour_idx = self._get_flatten_index(row, col)
                if neighbour_idx not in visited:
                    neighbour = self._create_node((col, row), cost, 
                                                  curr_node_idx)

                    if neighbour_idx not in queue or neighbour.cost_f < queue.cost_f_list[neighbour_idx]:
                        queue.push(neighbour)

                    # else:
                    #     if neighbour.cost_f < queue.cost_f_list[neighbour_idx]:
                    #         queue.update(neighbour)
