import numpy as np
from events import EventManager, Arrival, Departure
from workstations import Workstation_I1, Workstation_I2, Workstation_I3
from util import workstation_travel_time
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
        number_robots = 0.
        for ws in self.workstation_dict.values():
            ws.flow_stats(total_time=self.simulation_steps, output_result=False)
            try:
                number_robots += 1.*ws.flow_sim * ws.avg_sojourn_sim
            except:
                pass
            if ws.type=='I1':
                number_robots += 1.*ws.departure_count/self.simulation_steps*workstation_travel_time(ws, ws.assigned_workstation_I2)
        return number_robots


def exp1_change_rho(output_file_name = 'rho_result.csv'):
    import pandas as pd
    exp_record = []
    for rho2 in np.arange(0.01, 0.99, 0.01):
        print(rho2)
        ES2 = 100
        VarS2 = 16.
        special_pod_size = 100
        workstation_dict = {
            0: Workstation_I2(id=0, x=10, y=0, E_service_time=ES2, Var_service_time=VarS2, service_distribution='norm'),
        }
        departure_rate_I1 = rho2 / ES2
        arrival_rate_I1 = departure_rate_I1 * special_pod_size
        rho1 = 0.5
        ES1 = rho1 / arrival_rate_I1
        VarS1 = (ES1 / 3.) ** 2.
        # import pdb; pdb.set_trace()
        workstation_dict[1] = Workstation_I1(id=1, x=0, y=10, E_service_time=ES1,
                                             Var_service_time=VarS1, special_pod_size=special_pod_size,
                                             assigned_workstation_I2=workstation_dict[0], service_distribution='norm')
        arrival_rate_dict = {1: arrival_rate_I1}
        simulation_steps = 10000 * ES2
        simulator = Simulator(workstation_dict, arrival_rate_dict, simulation_steps)
        simulator.run_simulation()
        for ws in workstation_dict.values():
            ws.flow_stats(total_time=simulation_steps)

        #error_predict_theorem = 1./special_pod_size*rho2*(1.+abs(VarS1/ES1/ES1-1))/(2.*(1.-rho2)+rho2*VarS2/ES2/ES2)
        error_predict_theorem = rho2/2./special_pod_size/(1.-rho2)*(1.+abs(VarS1/ES1/ES1-1.))
        exp_record.append([ES1, ES2, arrival_rate_I1, rho1, rho2, workstation_dict[0].sojourn_theory_pure,
                           workstation_dict[0].sojourn_theory, workstation_dict[0].avg_sojourn_sim,
                           workstation_dict[0].cv2_inter_arrival_simulation, error_predict_theorem])
    exp_result = pd.DataFrame(exp_record,
                              columns=['ES1', 'ES2', 'arrival rate I1', 'rho1', 'rho2', 'approximated sojourn time',
                                       'Kingman formula sojourn time', 'simulation sojourn time',
                                       'CV of inter-arrival at I2', 'error predict theorem'])
    exp_result.to_csv(output_file_name)

def exp1_change_capacity(output_file_name='capacity_result.csv'):
    import pandas as pd
    exp_record = []
    rho2 = 0.2
    for special_pod_size in range(20, 200):
        print(special_pod_size)
        special_pod_size = int(special_pod_size)
        ES2 = 100
        VarS2 = 16.
        workstation_dict = {
            0: Workstation_I2(id=0, x=10, y=0, E_service_time=ES2, Var_service_time=VarS2, service_distribution='norm'),
        }
        departure_rate_I1 = rho2 / ES2
        arrival_rate_I1 = departure_rate_I1 * special_pod_size
        rho1 = 0.5
        ES1 = rho1 / arrival_rate_I1
        VarS1 = (ES1 / 3.) ** 2.
        # import pdb; pdb.set_trace()
        workstation_dict[1] = Workstation_I1(id=1, x=0, y=10, E_service_time=ES1,
                                             Var_service_time=VarS1, special_pod_size=special_pod_size,
                                             assigned_workstation_I2=workstation_dict[0], service_distribution='norm')
        arrival_rate_dict = {1: arrival_rate_I1}
        simulation_steps = 10000 * ES2
        simulator = Simulator(workstation_dict, arrival_rate_dict, simulation_steps)
        simulator.run_simulation()
        for ws in workstation_dict.values():
            ws.flow_stats(total_time=simulation_steps)
        error_predict_theorem = rho2 / 2. / special_pod_size / (1. - rho2) * (1. + abs(VarS1 / ES1 / ES1 - 1.))
        exp_record.append([ES1, ES2, arrival_rate_I1, rho1, rho2, workstation_dict[0].sojourn_theory_pure,
                           workstation_dict[0].sojourn_theory, workstation_dict[0].avg_sojourn_sim,
                           workstation_dict[0].cv2_inter_arrival_simulation, special_pod_size, error_predict_theorem])
    exp_result = pd.DataFrame(exp_record,
                              columns=['ES1', 'ES2', 'arrival rate I1', 'rho1', 'rho2', 'approximated sojourn time',
                                       'Kingman formula sojourn time', 'simulation sojourn time',
                                       'CV of inter-arrival at I2','special pod capacity','error_predict_theorem'])
    exp_result.to_csv(output_file_name)

# test simulator:
if __name__ == '__main__':
    exp1_change_rho()
    exp1_change_capacity()
    #workstation_dict[0].plot_queue()
