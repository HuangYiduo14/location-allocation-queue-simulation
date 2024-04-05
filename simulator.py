import numpy as np
from events import EventManager, Arrival, Departure

class Simulator:
    def __init__(self, workstation_list, arrival_rate_dict, simulation_steps): # arrival_rate_dict = {station_id: arrival rate,...}
        self.time = 0
        self.workstation_list = workstation_list
        self.arrival_rate = arrival_rate_dict
        self.simulation_steps = simulation_steps
        self.I1 = []
        self.I2 = []
        self.I3 = []
        for ws in workstation_list:
            if ws.type=='I1':
                self.I1.append(ws.id)
            elif ws.type == 'I2':
                self.I2.append(ws.id)
            elif ws.type == 'I3':
                self.I3.append(ws.id)
            else:
                raise ValueError('unknown workstation type')
        self.event_manager = EventManager()
        self.generate_new_arrivals()


    def generate_new_arrivals_one_station(self, i):
        self.event_manager.addevent(Arrival(self.time + np.random.exponential(1. / self.arrival_rate[i]), 'normal', i))

    def generate_new_arrivals(self):
        for i in self.I1:
            self.generate_new_arrivals_one_station(i)
        for i in self.I3:
            self.generate_new_arrivals_one_station(i)

    def forward_time(self, time_diff):
        for ws in self.workstation_list:
            ws.remain_busy_time = max(0., ws.remain_busy_time - time_diff)

    def run_simulation(self):
        while self.time < self.simulation_steps:
            event = self.event_manager.getevent()
            event_time = event.time
            time_diff = event_time - self.time
            # update remaining busy time in each ws
            self.forward_time(time_diff)
            self.time = self.time + time_diff
            # simulate events
            if event.type=='arrival':
                next_event = self.workstation_list[event.workstation].customer_arrive(event)
                if event.workstation.type in ['I1', 'I3']:
                    self.generate_new_arrivals_one_station(event.workstation.id)
            elif event.type=='departure':
                next_event = self.workstation_list[event.workstation].customer_departure(event)
            else:
                raise ValueError('event type nonexist')
            if isinstance(next_event, int):
                pass # if the returned a int, no new event
            else:
                self.event_manager.addevent(next_event)

