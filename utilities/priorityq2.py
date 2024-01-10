import heapq

class PriorityQueue:
    def __init__(self, heap=None):
        self.heap = [] if heap is None else heap
        self.index = 0

    def put(self, item):
        heapq.heappush(self.heap, item)

    def get(self):
        return heapq.heappop(self.heap)

    def put_with_priority(self, priority_tuple, item):
        entry = (priority_tuple, self.index, item)
        heapq.heappush(self.heap, entry)
        self.index += 1

    def get_with_priority(self):
        if self.heap:
            priority, index, item = heapq.heappop(self.heap)
            return priority[0], item  # Return priority tuple components separately
        else:
            return None

    def copy(self):
        return PriorityQueue(self.heap.copy())

    def __len__(self):
        return len(self.heap)
