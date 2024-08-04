import csv
import heapq
edgeFile = 'edges.csv'


def ucs(start, end):
    # Begin your code (Part 3)
    """
    Part I: Read the 'edges.csv'
    1. Create a dictionary named 'edges' to store the graph.
    2. Use csv.reader() to read the whole file.
    3. Since the first row is the column, use next() to skip the first one.
    4. Since every node should be a key in the dictionary, for every node in the file,
       create a key in the 'edges' for it.
    5. Store each path which (key = start ID) and (value = (end ID, distance)) in 'edges'.
    """
    
    edges = {}
    with open(edgeFile) as file:
        csvfile = csv.reader(file)
        next(csvfile)
        for row in csvfile:
            first_node = int(row[0])
            second_node = int(row[1])
            distance = float(row[2])
            if first_node not in edges:
                edges[first_node] = []
            if second_node not in edges:
                edges[second_node] = []
            edges[first_node].append((second_node, distance))
    
    """
    Part II: UCS
    1. Create some variables:
        (a) path_visited: a dictionary for storing the distance between visited nodes.
        (b) priority_queue: a heap storing tuples of (cost, node) to prioritize exploring nodes.
        (c) visited: a set storing the nodes which have been visited.
        (d) num_visited: record the number of visited node for return.
    2. Push the start node with cost 0 to the 'priority_queue'.
    3. While the 'priority_queue' is not empty, pop the node with the least cost.
    4. If the popped node is the endpoint, compute the path.
       If not, traverse its neighbors. If its neighbor isn't in the 'visited', i.e., hasn't been 
       visited, add this node and its distance to 'priority_queue', add this node to 'visited', 
       and store this node and its distance to 'path_visited'.
    5. Compute the path:
        (a) Create some variables:
            (i)  path: a list storing the nodes on the path.
            (ii) dist: equal to the cost of the end point.
        (b) Add the 'end' to the 'path', since I want to compute path from the end point by 'path_visited'.
        (c) Repeat the below step until the first element of 'path' is the start point:
            (i)  Insert the neighbor of the current to the front of 'path'.
        (d) Return 'path', 'dist', and 'num_visited'.
    6. If UCS doesn't find the path successfully, return empty path, infinite, 'num_visited'.
    """
    path_visited = {}
    priority_queue = [(0, start)]
    visited = set()
    num_visited = 0
    while priority_queue:
        cost, node = heapq.heappop(priority_queue)
        if node == end:
            path = []
            dist = cost
            path.append(end)
            while path[0] != start:
                path.insert(0, path_visited[path[0]][0])
            return path, dist, num_visited
        if node not in visited:
            num_visited += 1
            visited.add(node)
            for end_node, dis in edges[node]:
                if end_node not in visited:
                    heapq.heappush(priority_queue, (cost + dis, end_node))
                    path_visited[end_node] = (node, dis)
    return [], float('inf'), num_visited  
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
