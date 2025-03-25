from itertools import permutations
import numpy as np


def tsp_brute_force_search(start_node, graph):
    possible_routes = permutations(set(graph.keys()) - {start_node})
    shortest_route = []
    minimum_cost = np.Inf
    for route in possible_routes:
        curr_node = start_node
        cost = 0.
        for next_node in route:
            cost += graph[curr_node][next_node]
            curr_node = next_node

        cost += graph[curr_node][start_node]
        if cost < minimum_cost:
            shortest_route = list(route)
            minimum_cost = cost

    shortest_route.insert(0, start_node)
    shortest_route.append(start_node)
    return shortest_route, minimum_cost


def tsp_pruned_search(start_node, graph, route_so_far=[], cost_so_far=0., minimum_cost=np.Inf):
    if not route_so_far:
        route_so_far = [start_node]

    shortest_route = []
    possible_next_nodes = set(graph.keys()) - set(route_so_far)
    curr_node = route_so_far[-1]

    if not possible_next_nodes:
        route = route_so_far.copy()
        route.append(start_node)
        cost = cost_so_far + graph[curr_node][start_node]
        return route, cost

    for next_node in possible_next_nodes:
        cost = cost_so_far + graph[curr_node][next_node]
        route = route_so_far.copy()
        route.append(next_node)
        if cost > minimum_cost:
            continue

        route, cost = tsp_pruned_search(start_node, graph, route, cost, minimum_cost)

        if cost < minimum_cost:
            shortest_route = route
            minimum_cost = cost

    return shortest_route, minimum_cost
