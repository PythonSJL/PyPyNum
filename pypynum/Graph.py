class BaseGraph:
    def __init__(self):
        self.graph = {}

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, sorted(self.graph.items()))

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = []

    def remove_vertex(self, vertex):
        if vertex in self.graph:
            neighbors = self.graph[vertex]
            del self.graph[vertex]
            for neighbor in neighbors:
                self.graph[neighbor].remove(vertex)

    def has_vertex(self, vertex):
        return vertex in self.graph

    def has_edge(self, vertex1, vertex2):
        return vertex1 in self.graph and vertex2 in self.graph[vertex1]

    def get_edges(self, vertex):
        return self.graph.get(vertex, {})

    def all_vertices(self):
        return sorted(set(list(self.graph) + sum([list(self.graph[vertex]) for vertex in self.graph], [])))

    def all_edges(self):
        edges = []
        for vertex, neighbours in self.graph.items():
            for neighbour in neighbours:
                edges.append((vertex, neighbour))
        return sorted(edges)

    def to_adjacency_matrix(self):
        vertices = self.all_vertices()
        num_vertices = len(vertices)
        matrix = [[0] * num_vertices for _ in range(num_vertices)]
        if isinstance(self, BaseWeGraph):
            for u in self.graph:
                for v in self.graph[u].items():
                    try:
                        v, w = v
                        matrix_index_u = vertices.index(u)
                        matrix_index_v = vertices.index(v)
                        matrix[matrix_index_u][matrix_index_v] = w
                    except ValueError:
                        continue
        else:
            for u in self.graph:
                for v in self.graph[u]:
                    try:
                        matrix_index_u = vertices.index(u)
                        matrix_index_v = vertices.index(v)
                        matrix[matrix_index_u][matrix_index_v] = 1
                    except ValueError:
                        continue
        return [[self.__class__.__name__] + vertices] + [[vertex] + row for vertex, row in zip(vertices, matrix)]

    def dfs(self, start_vertex, visited=None):
        if visited is None:
            visited = set()
        visited.add(start_vertex)
        search_list = [start_vertex]
        if start_vertex not in self.graph:
            return [start_vertex]
        for neighbour in self.graph[start_vertex]:
            if neighbour not in visited:
                search_list.extend(self.dfs(neighbour, visited))
        return search_list

    def bfs(self, start_vertex):
        visited = {start_vertex}
        queue = [start_vertex]
        search_list = [start_vertex]
        while queue:
            vertex = queue.pop(0)
            if vertex in self.graph:
                for neighbour in self.graph[vertex]:
                    if neighbour not in visited:
                        visited.add(neighbour)
                        queue.append(neighbour)
                        search_list.append(neighbour)
        return search_list

    def is_connected(self):
        visited = set()
        queue = []
        for node in self.graph:
            if node not in visited:
                visited.add(node)
                queue.append(node)
                break
        while queue:
            node = queue.pop(0)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(self.graph)

    def is_complete(self):
        num_nodes = len(self.graph)
        return all([len(self.graph[node]) == num_nodes - 1 for node in self.graph])


class DiGraph(BaseGraph):
    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.graph:
            self.graph[vertex1].append(vertex2)
        else:
            self.graph[vertex1] = [vertex2]

    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.graph and vertex2 in self.graph[vertex1]:
            self.graph[vertex1].remove(vertex2)
            if not self.graph[vertex1]:
                del self.graph[vertex1]

    def __str__(self):
        edges = []
        for vertex, neighbours in self.graph.items():
            for neighbour in neighbours:
                edges.append("{} -> {}".format(vertex, neighbour))
        return "\n".join(sorted(edges))


class UnGraph(BaseGraph):
    def add_edge(self, vertex1, vertex2):
        if vertex1 in self.graph:
            self.graph[vertex1].append(vertex2)
        else:
            self.graph[vertex1] = [vertex2]
        if vertex2 in self.graph:
            self.graph[vertex2].append(vertex1)
        else:
            self.graph[vertex2] = [vertex1]

    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.graph and vertex2 in self.graph:
            if vertex2 in self.graph[vertex1]:
                self.graph[vertex1].remove(vertex2)
            if vertex1 in self.graph[vertex2]:
                self.graph[vertex2].remove(vertex1)
            if not self.graph[vertex1]:
                del self.graph[vertex1]
            if not self.graph[vertex2]:
                del self.graph[vertex2]

    def __str__(self):
        edges = []
        for vertex, neighbours in self.graph.items():
            for neighbour in neighbours:
                edges.append("{} -- {}".format(vertex, neighbour))
        return "\n".join(sorted(edges))


class BaseWeGraph(BaseGraph):
    def add_edge(self, vertex1, vertex2, weight=0):
        raise NotImplementedError

    def remove_edge(self, vertex1, vertex2):
        raise NotImplementedError

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = {}

    def remove_vertex(self, vertex):
        if vertex in self.graph:
            neighbors = self.graph[vertex]
            del self.graph[vertex]
            for neighbor in neighbors:
                self.remove_edge(neighbor, vertex)

    def get_edge_weight(self, vertex1, vertex2):
        if self.has_edge(vertex1, vertex2):
            return self.graph[vertex1][vertex2]

    def get_in_degree_weight_sum(self, vertex):
        return sum([self.get_edge_weight(neighbor, vertex)
                    for neighbor in self.graph if vertex in self.graph[neighbor]])

    def get_out_degree_weight_sum(self, vertex):
        return sum([self.get_edge_weight(vertex, neighbor) for neighbor in self.graph.get(vertex, {})])

    def dijkstra(self, start):
        distances = {vertex: float("inf") for vertex in self.graph}
        distances[start] = 0
        queue = [(0, start)]
        visited = set()
        while queue:
            current_distance, current_vertex = min(queue, key=lambda x: x[0])
            if current_vertex in visited:
                queue.remove((current_distance, current_vertex))
                continue
            visited.add(current_vertex)
            queue.remove((current_distance, current_vertex))
            if current_distance > distances[current_vertex] or current_vertex not in self.graph:
                continue
            for neighbor, weight in self.graph[current_vertex].items():
                distance = current_distance + weight
                if neighbor in distances:
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        queue.append((distance, neighbor))
                else:
                    distances[neighbor] = distance
                    queue.append((distance, neighbor))
        return distances

    def reconstruct_path(self, start, end, distances):
        current = end
        path = [current]
        while current != start:
            for neighbor, weight in self.get_edges(current).items():
                if distances[current] == distances[neighbor] + weight:
                    current = neighbor
                    path.append(current)
                    break
        return path[::-1]


class WeDiGraph(BaseWeGraph):
    def add_edge(self, vertex1, vertex2, weight=0):
        if vertex1 not in self.graph:
            self.graph[vertex1] = {}
        self.graph[vertex1][vertex2] = weight

    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.graph and vertex2 in self.graph[vertex1]:
            del self.graph[vertex1][vertex2]
            if not self.graph[vertex1]:
                del self.graph[vertex1]

    def __str__(self):
        edges = []
        for vertex, neighbours in self.graph.items():
            for neighbour, weight in neighbours.items():
                edges.append("{} -[{}]> {}".format(vertex, weight, neighbour))
        return "\n".join(sorted(edges))


class WeUnGraph(BaseWeGraph):
    def add_edge(self, vertex1, vertex2, weight=0):
        if vertex1 not in self.graph:
            self.graph[vertex1] = {}
        if vertex2 not in self.graph:
            self.graph[vertex2] = {}
        self.graph[vertex1][vertex2] = weight
        self.graph[vertex2][vertex1] = weight

    def remove_edge(self, vertex1, vertex2):
        if vertex1 in self.graph and vertex2 in self.graph[vertex1]:
            del self.graph[vertex1][vertex2]
            del self.graph[vertex2][vertex1]
            if not self.graph[vertex1]:
                del self.graph[vertex1]
            if vertex1 != vertex2 and not self.graph[vertex2]:
                del self.graph[vertex2]

    def __str__(self):
        edges = []
        for vertex, neighbours in self.graph.items():
            for neighbour, weight in neighbours.items():
                edges.append("{} -[{}]- {}".format(vertex, weight, neighbour))
        return "\n".join(sorted(edges))
