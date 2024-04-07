import numpy as np
from events import Arrival, Departure
from util import workstation_travel_time
import matplotlib.pyplot as plt


# workstaion: I1, I2, and I3
class WorkStation:
    def __init__(self, id, x, y, E_service_time, Var_service_time, service_distribution='exp'):
        self.id = id
        self.x = x
        self.y = y
        self.E_service_time = E_service_time
        self.Var_service_time = Var_service_time
        self.type = 'unknown'
        self.waiting_customers = 0
        self.remain_busy_time = 0
        self.service_distribution = service_distribution
        # record arrivals and departures
        self.arrival_count = 0
        self.departure_count = 0
        # record arrival time and departure time
        self.arrival_time_list = []
        self.departure_time_list = []
        # record queue length
        self.queue_length_when_arrival = []
        self.queue_length_when_departure = []
        self.queue_length_when_event = []
        self.event_time = []

    def record_arrival(self, arrival_time):
        # record arrival time and event
        self.arrival_count += 1
        self.arrival_time_list.append(arrival_time)
        self.queue_length_when_arrival.append(self.waiting_customers)
        self.event_time.append(arrival_time)
        self.queue_length_when_event.append(self.waiting_customers)
        self.event_time.append(arrival_time)
        self.queue_length_when_event.append(self.waiting_customers + 1)

    def record_departure(self, departure_time):
        # record departure time and event
        self.departure_count += 1
        self.departure_time_list.append(departure_time)
        self.queue_length_when_departure.append(self.waiting_customers)
        self.event_time.append(departure_time)
        self.queue_length_when_event.append(self.waiting_customers)
        self.event_time.append(departure_time)
        self.queue_length_when_event.append(self.waiting_customers - 1)

    def __repr__(self):
        return self.id, self.type

    def generate_service_time(self):
        if self.service_distribution == 'exponential' or self.service_distribution == 'exp':
            return np.random.exponential(scale=self.E_service_time)
        elif self.service_distribution == 'normal' or self.service_distribution == 'norm':
            return max(0., self.E_service_time + np.sqrt(
                self.Var_service_time) * np.random.normal())  # we must truncate the norma
        else:
            raise ValueError('unknown service time distribution')

    def customer_arrive(self, arrival_event: Arrival):
        arrival_time = arrival_event.time
        token_type = arrival_event.token_type
        # update queue info
        self.record_arrival(arrival_time)
        self.waiting_customers += 1
        # create a departure new event
        service_time = self.generate_service_time()
        departure_time = arrival_time + self.remain_busy_time + service_time
        self.remain_busy_time += service_time
        new_departure = Departure(departure_time, token_type, workstation=self)
        return [new_departure]

    def customer_departure(self, departure_event: Departure):
        assert self.waiting_customers > 0
        departure_time = departure_event.time
        # record arrival time and event
        self.record_departure(departure_time)
        self.waiting_customers -= 1
        return []

    def flow_stats(self, total_time, cv2_arrival=1.):
        time_array = np.array(self.event_time)
        queue_array = np.array(self.queue_length_when_event)
        time_diff = time_array[1:] - time_array[:-1]
        queue_length = queue_array[1:]
        total_sojourn = np.dot(time_diff, queue_length)
        avg_sojourn_sim = total_sojourn / self.arrival_count
        print('avg sojourn time simulation: ', avg_sojourn_sim)
        flow = self.arrival_count / total_time
        rho = flow * self.E_service_time
        assert rho < 1.
        cv2_service_time = self.Var_service_time / self.E_service_time / self.E_service_time
        sojourn_theory = rho / (1. - rho) * (
                    cv2_service_time + cv2_arrival) / 2. * self.E_service_time + self.E_service_time
        print('sojourn time theory', sojourn_theory)
        print('workstation id:', self.id)
        print('number of arrivals', self.arrival_count)
        print('number of departures', self.departure_count)

    def plot_queue(self):
        plt.figure()
        plt.plot(self.event_time, self.queue_length_when_event)
        plt.show()
        plt.savefig('queue over time.png')


class Workstation_I2(WorkStation):
    def __init__(self, id, x, y, E_service_time, Var_service_time, service_distribution):
        super().__init__(id, x, y, E_service_time, Var_service_time, service_distribution)
        self.type = 'I2'


class Workstation_I3(WorkStation):
    def __init__(self, id, x, y, E_service_time, Var_service_time, service_distribution):
        super().__init__(id, x, y, E_service_time, Var_service_time, service_distribution)
        self.type = 'I3'


class Workstation_I1(WorkStation):

    def __init__(self, id, x, y, E_service_time, Var_service_time, special_pod_size: int,
                 assigned_workstation_I2: WorkStation, service_distribution):
        super().__init__(id, x, y, E_service_time, Var_service_time, service_distribution)
        self.type = 'I1'
        self.special_pod_size = special_pod_size
        self.assigned_workstation_I2 = assigned_workstation_I2
        self.special_pod_inventory_level = 0
        self.special_departure_count = 0
        self.special_departure_time_list = []

    def customer_departure(self, departure_event: Departure):
        super(Workstation_I1, self).customer_departure(departure_event)
        self.special_pod_inventory_level += 1
        if self.special_pod_inventory_level == self.special_pod_size:
            departure_time = departure_event.time
            arrival_time_at_I2 = departure_time + workstation_travel_time(self, self.assigned_workstation_I2)
            new_arrival_event = Arrival(arrival_time_at_I2, 'special', self.assigned_workstation_I2)
            self.special_pod_inventory_level = 0
            self.special_departure_count += 1
            self.special_departure_time_list.append(departure_time)
            return [new_arrival_event]
        else:
            return []

    def flow_stats(self, total_time, cv2_arrival=1.):
        super(Workstation_I1, self).flow_stats(total_time, cv2_arrival)
        print('special pod departures', self.special_departure_count)
