#!/usr/bin/python

import itertools
from heapq import heappush, heappop

REMOVED = '<removed-task>'  # placeholder for a removed task


class PriorityQueue:

    def __init__(self):
        self.pq = []  # list of entries arranged in a heap
        self.entry_finder = {}  # mapping of tasks to entries
        self.counter = itertools.count()  # unique sequence count

    def add(self, task, priority=0):
        'Add a new task or update the priority of an existing task'
        if task in self.entry_finder:
            self.remove(task)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)

    def remove(self, task):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        if task in self.entry_finder:
            entry = self.entry_finder.pop(task)
            entry[-1] = REMOVED
            return entry[0]
        else:
            # Gestione della situazione in cui il task non è presente
            raise KeyError(f'Task {task} not found in priority queue')

    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self.pq:
            priority, count, task = heappop(self.pq)
            if task != REMOVED:
                del self.entry_finder[task]
                return task
        raise KeyError('pop from an empty priority queue')
    
    def __len__(self):
        return len(self.pq)

    @classmethod    
    def merge_queues(cls, *queues):
        merged = cls()
        for queue in queues:
            for entry in queue.pq:
                merged.add(entry[2], entry[0])
        return merged