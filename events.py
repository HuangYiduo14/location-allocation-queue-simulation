import heapq as hq
class EventManager:
    def __init__(self):
        self.event_queue = []
        hq.heapify(self.event_queue)

    def addevent(self, event):
        hq.heappush(self.event_queue, event)

    def getevent(self):
        return hq.heappop(self.event_queue)

class Event:
    def __init__(self, time, token_type, workstation):
        self.time = time
        self.token_type = token_type
        self.workstation = workstation
        self.type = 'unknown'

    def __lt__(self, event2):
        return self.time < event2.time

    def __gt__(self, event2):
        return self.time > event2.time

    def __repr__(self):
        return self.type, self.time, self.token_type, self.workstation

class Arrival(Event):
    def __init__(self, time, token_type, workstation):
        super().__init__(time, token_type, workstation)
        self.type = 'arrival'

class Departure(Event):
    def __init__(self, time, token_type, workstation):
        super().__init__(time, token_type, workstation)
        self.type = 'departure'