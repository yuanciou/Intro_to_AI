import csv
edgeFile = 'edges.csv'


def bfs(start, end):
    # Begin your code (Part 1)
    """
    Part I: Read the 'edges.csv'
    1. Create a dictionary named 'edges' to store the graph.
    2. Use csv.reader() to read the whole file.
    3. Since the first row is the column, use next() to skip the first one.
    4. Since every node should be a key in the dictionary, for every node in the file,
       create a key in the 'edges' for it.
    5. Store each path which (key = start ID) and (value = (end ID, distince)) in 'edges'.
    """
    
    edges = {}
    with open(edgeFile) as file:
        csvfile = csv.reader(file)
        next(csvfile)
        for row in csvfile:
            first_node = int(row[0])
            second_node = int(row[1])
            distance = float(row[2])
            if (first_node not in edges):
                edges[first_node] = []
            if (second_node not in edges):
                edges[second_node] = []
            edges[first_node].append((second_node, distance))
    
    """
    Part II: BFS
    1. Create some variables:
        (a) path_visited: a dictionary for storing the distance between visited nodes.
        (b) queue: a list storing pair for the neighbor and its distance to current nodes.
        (c) visited: a set storing the nodes which have been visited.
        (d) num_visited: record the number of visited node for return.
    2. Pop the first element of 'queue'.
    3. If the node pop from the queue is the endpoint of whole function, compute the path.
       If not, plus one to the 'num_visited', and traverse its neighbors. If its neighbor isn't in
       the 'visited', i.e, hasn't been visited, add this node and its distance to 'queue', add this
       node to 'visited', and store this node and its distance to 'path_visited'.
    4. Compute the path:
        (a) Create some variables:
            (i)  path: a list storing the nodes on the path.
            (ii) dist: count the total distance of the path.
        (b) Add the 'end' to the 'path', since I want to compute path from the end point by 'path_visited'.
        (c) Repeat the below step until the fist element of 'path' is the start point:
            (i)  Use 'dist' to plus the distance between current node and its neighbor on the path.
            (ii) Insert the neighbor of the current to the front of 'path'.
        (d) Return 'path', 'dist', and 'num_visited'.
    5. If bfs doesn't find the path successfully, return empty path, infinite, 'num_visited'.
    """
    path_visited = {}
    queue = [(start, 0)]
    visited = set()
    visited.add(start)
    num_visited = 0
    while queue:
        node = queue.pop(0)
        if node[0] == end:
            path = []
            dist = 0
            path.append(end)
            while path[0] != start:
                dist = dist + path_visited[path[0]][1]
                path.insert(0, path_visited[path[0]][0])
            return path, dist, num_visited
        else:
            num_visited += 1
            for end_node, dis in edges[node[0]]:
                if end_node not in visited:
                    queue.append((end_node, dis))
                    visited.add(end_node)
                    path_visited[end_node] = (node[0], dis)
    return [], float('inf'), num_visited    
    # End your code (Part 1)


if __name__ == '__main__':
    path, dist, num_visited = bfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
