import heapq

class PriorityQueue:
    def __init__(self, heap=None):
        self.heap = [] if heap is None else heap
        self.index = 0

    def put(self, item):
        heapq.heappush(self.heap, item)

    def get(self):
        return heapq.heappop(self.heap)

    def put_with_priority(self, priority, item):
        self.index += 1
        heapq.heappush(self.heap, (priority, self.index, item))

    def get_with_priority(self):
        return heapq.heappop(self.heap) if len(self.heap) > 0 else None
    
    def copy(self):
        return PriorityQueue(self.heap.copy())
    
    def __len__(self):
        return len(self.heap)