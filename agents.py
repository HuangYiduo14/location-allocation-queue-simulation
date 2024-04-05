import numpy as np
from events import Arrival, Departure
from util import workstation_travel_time
# workstaion: I1, I2, and I3
class WorkStation:
    def __init__(self, id, x, y, E_service_time, Var_service_time):
        self.id = id
        self.x = x
        self.y = y
        self.E_service_time = E_service_time
        self.Var_service_time = Var_service_time
        self.type = 'unknown'
        self.waiting_customers = 0
        self.remain_busy_time = 0

    def __repr__(self):
        return self.id, self.type

    def generate_service_time(self):
        return np.random.exponential(scale=1./self.E_service_time)

    def customer_arrive(self, arrival_event: Arrival):
        arrival_time = arrival_event.time
        token_type = arrival_event.token_type
        # update queue info
        self.waiting_customers += 1
        # create a departure new event
        service_time = self.generate_service_time()
        departure_time = arrival_time + self.remain_busy_time + service_time
        self.remain_busy_time += service_time
        new_departure = Departure(departure_time, token_type, workstation=self.id)
        return new_departure

    def customer_departure(self, departure_event: Departure):
        assert self.waiting_customers>0
        self.waiting_customers -= 1
        return 0


class Workstation_I1(WorkStation):
    def __init__(self, id, x, y, E_service_time, Var_service_time, special_pod_size, assigned_workstation_I2):
        super().__init__(id, x, y, E_service_time, Var_service_time)
        self.type = 'I1'
        self.pool_buffer = special_pod_size
        self.pool_inventory_level = 0
        self.assigned_workstation_I2 = assigned_workstation_I2

    def customer_arrive(self, arrival_event: Arrival):
        arrival_time = arrival_event.time
        token_type = arrival_event.token_type
        # update queue info
        self.waiting_customers += 1
        # create a departure new event
        service_time = self.generate_service_time()
        departure_time_to_inventory = arrival_time + self.remain_busy_time + service_time
        self.remain_busy_time += service_time
        self.pool_inventory_level += 1
        if self.pool_inventory_level == self.pool_buffer: # if the special pod is loaded after this arrival, create a departure event
            self.pool_inventory_level = 0
            return Departure(departure_time_to_inventory, 'special', workstation=self.id)
        else:
            return 0

    def customer_departure(self, departure_event: Departure):
        departure_time = departure_event.time
        assert self.waiting_customers > 0
        self.waiting_customers -= 1
        arrival_time_at_I2 = departure_time + workstation_travel_time(self, self.assigned_workstation_I2)
        new_arrival = Arrival(arrival_time_at_I2, 'special', self.assigned_workstation_I2.id)
        return new_arrival


class Workstation_I2(WorkStation):
    def __init__(self, id, x, y, E_service_time, Var_service_time):
        super().__init__(id, x, y, E_service_time, Var_service_time)
        self.type = 'I2'

class Workstation_I3(WorkStation):
    def __init__(self, id, x, y, E_service_time, Var_service_time):
        super().__init__(id, x, y, E_service_time, Var_service_time)
        self.type = 'I3'
