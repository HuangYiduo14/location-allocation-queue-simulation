import numpy as np
from events import EventManager, Arrival, Departure
from workstations import Workstation_I1, Workstation_I2, Workstation_I3

class Simulator:
    def __init__(self, workstation_dict, arrival_rate_dict,
                 simulation_steps):  # arrival_rate_dict = {station_id: arrival rate,...}
        # workstation_dict = {ws_id: WS,...}
        self.time = 0
        self.workstation_dict = workstation_dict
        self.arrival_rate = arrival_rate_dict
        self.simulation_steps = simulation_steps
        self.I1 = []
        self.I2 = []
        self.I3 = []
        for ws in workstation_dict.values():
            if ws.type == 'I1':
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
        self.event_manager.addevent(
            Arrival(self.time + np.random.exponential(1. / self.arrival_rate[i]), 'normal', self.workstation_dict[i]))

    def generate_new_arrivals(self):
        for i in self.I1:
            self.generate_new_arrivals_one_station(i)
        for i in self.I3:
            self.generate_new_arrivals_one_station(i)

    def forward_time(self, time_diff):
        for ws in self.workstation_dict.values():
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
            if event.type == 'arrival':
                next_events = self.workstation_dict[event.workstation.id].customer_arrive(event)
                if event.workstation.type in ['I1', 'I3']:
                    self.generate_new_arrivals_one_station(event.workstation.id)
            elif event.type == 'departure':
                next_events = event.workstation.customer_departure(event)
            else:
                raise ValueError('event type nonexist')
            for next_event in next_events:
                self.event_manager.addevent(next_event)
        number_robots = 0
        for ws in self.workstation_dict.values():
            ws.flow_stats(total_time=self.simulation_steps, output_result=False)
            try:
                number_robots += ws.flow_sim * ws.avg_sojourn_sim
            except:
                pass
        return number_robots


# test simulator:
if __name__ == '__main__':
    workstation_dict = {
        0: Workstation_I3(id=0, x=0, y=0, E_service_time=10, Var_service_time=4, service_distribution='norm'),
        1: Workstation_I2(id=1, x=10, y=0, E_service_time=100, Var_service_time=4, service_distribution='norm'),
        2: Workstation_I2(id=2, x=10, y=10, E_service_time=100, Var_service_time=4, service_distribution='norm'),
    }
    workstation_dict[3] = Workstation_I1(id=3, x=0, y=10, E_service_time=10,
                                           Var_service_time=4, special_pod_size=100,
                                           assigned_workstation_I2=workstation_dict[1], service_distribution='norm')
    workstation_dict[4] = Workstation_I1(id=4, x=0, y=20, E_service_time=10,
                                           Var_service_time=4, special_pod_size=100,
                                           assigned_workstation_I2=workstation_dict[2], service_distribution='norm')
    arrival_rate_dict = {0: 0.05, 3: 0.05, 4: 0.05}
    simulation_steps = 100000.
    simulator = Simulator(workstation_dict, arrival_rate_dict, simulation_steps)
    simulator.run_simulation()
    for ws in workstation_dict.values():
        ws.flow_stats(total_time=simulation_steps)
    #workstation_dict[0].plot_queue()
