from workstations import Workstation_I1, Workstation_I2, Workstation_I3
from util import EPS
from simulator import Simulator
import numpy as np

class Warehouse:
    def __init__(self, width, height, demand_density, min_distance, X, E_S1, Var_S1, E_S2, Var_S2, special_pod_size, has_XYZ = False, Y=None, Z=None):
        self.width = width
        self.height = height
        # create the cells demand
        self.demand_map = demand_density * np.ones((width, height))
        self.workstation_dict = dict()
        self.workstation_id = 0
        self.E_S1 = E_S1
        self.E_S2 = E_S2
        self.Var_S1 = Var_S1
        self.Var_S2 = Var_S2
        self.special_pod_size = special_pod_size
        self.open_I1 = []
        self.open_I2 = []
        self.open_I3 = []
        self.J_xy_dict = dict()
        self.xy_J_dict = dict()
        self.J_arrival_rate = dict()
        self.arrival_rate_dict = dict()
        self.generate_cells()
        I1_assigned_I2 = self.generate_workstation_candidates(min_distance)

        if has_XYZ:
            self.load_X(X)
            self.load_YZ(Y,Z)
        else:
            self.load_X(X)
            Y,Z = self.genearte_YZ()
            self.load_YZ(Y,Z)
            print(X)
            print(Y)
            print(Z)

    def generate_cells(self):
        j = 0
        for x in range(self.width):
            for y in range(self.height):
                self.J_xy_dict[j] = (x,y)
                self.xy_J_dict[(x,y)] = j
                self.J_arrival_rate[j] = self.demand_map[x,y]
                j+=1

    def generate_workstation_candidates(self, min_distance):
        # we will genrated workstation candidates
        # I2 I3 are uniformly on the boundary, I1 are uniformly in the center
        # I1 stations are assigned to closest I2, the assigned result will be returned as a dict {I1_ws_id: assigned_I2_ws_id,...}
        min_distance = np.round(min_distance)
        assert min_distance<self.width//2
        assert min_distance<self.height//2
        n_X = self.width//min_distance + 1
        n_Y = self.height//min_distance + 1
        for i in range(n_X):# create I2 and I3
            for j in range(n_Y):
                if i==0 or j==0 or i==n_X-1 or j==n_Y-1:
                    if ((i==0 or i==n_X-1) and j%2==0) or ((j==0 or j==n_Y-1) and i%2==0): # create I3
                        self.workstation_dict[self.workstation_id] = Workstation_I3(id=self.workstation_id,
                                           x=1.*i*min_distance, y=1.*j*min_distance,
                                           E_service_time=self.E_S1, Var_service_time=self.Var_S1,
                                           service_distribution='norm')
                        self.workstation_id+=1
                    else: # create I2
                        self.workstation_dict[self.workstation_id] = Workstation_I2(id=self.workstation_id,
                                           x=1. * i * min_distance, y=1. * j * min_distance,
                                           E_service_time=self.E_S2, Var_service_time=self.Var_S2,
                                           service_distribution='norm')
                        self.workstation_id += 1
        I1_assigned_I2 = dict()
        for i in range(1,n_X-1):
            for j in range(1,n_Y-1):
                x= 1. * i * min_distance
                y= 1. * j * min_distance
                I2_distance_dict ={ws.id: abs(x-ws.x)+abs(y-ws.y) for ws in self.workstation_dict.values() if ws.type=='I2'}
                min_ws_I2 = min(I2_distance_dict.items(), key=lambda z: z[1])
                min_ws_I2_id = min_ws_I2[0]
                I1_assigned_I2[self.workstation_id] = min_ws_I2_id
                self.workstation_dict[self.workstation_id] = Workstation_I1(id=self.workstation_id,
                                    x=1. * i * min_distance, y=1. * j * min_distance,
                                    E_service_time=self.E_S1, Var_service_time=self.Var_S1,
                                    service_distribution='norm', special_pod_size=self.special_pod_size,
                                    assigned_workstation_I2=self.workstation_dict[min_ws_I2_id])
                self.workstation_id += 1
        return I1_assigned_I2

    def load_X(self, X):
        for key, ws in self.workstation_dict.items():
            if X[key]<EPS:
                del self.workstation_dict[key]
        self.open_I1 = [ws.id for ws in self.workstation_dict.values() if ws.type=='I1' and X[ws.id] > 1 - EPS]
        self.open_I2 = [ws.id for ws in self.workstation_dict.values() if ws.type=='I2' and X[ws.id] > 1 - EPS]
        self.open_I3 = [ws.id for ws in self.workstation_dict.values() if ws.type=='I3' and X[ws.id] > 1 - EPS]

    def genearte_YZ(self):
        # I1_assigned_I2 is a dict: {id_I1: id_I2,...}
        Y = dict() # assign normal pods to closest ws in I1 or I3
        Q_I =  {ws.id: 0.  for ws in self.workstation_dict.values()}
        for j in self.J_xy_dict.keys():
            x, y = self.J_xy_dict[j]
            I_distance_dict = {ws.id: abs(x - ws.x) + abs(y - ws.y) for ws in self.workstation_dict.values() if ws.type != 'I2'}
            min_ws_I = min(I_distance_dict.items(), key=lambda z: z[1])
            min_ws_I_id = min_ws_I[0]
            if (j,min_ws_I_id) in Y.keys():
                Y[(j, min_ws_I_id)] += self.J_arrival_rate[j]
            else:
                Y[(j, min_ws_I_id)] = self.J_arrival_rate[j]
            Q_I[min_ws_I_id] += self.J_arrival_rate[j]
        Z = dict()
        for id, q in Q_I.items():
            if id in self.open_I1:
                i = self.workstation_dict[id].assigned_workstation_I2.id
                k = id
                xi = self.workstation_dict[id].special_pod_size
                if (k,i) in Z.keys():
                    Z[(k,i)] += Q_I[k]/xi
                else:
                    Z[(k,i)] = Q_I[k]/xi
        return Y, Z

    def load_YZ(self, Y: dict, Z: dict):
        # Y[j,i] is the flow from cell j to i
        # Z[k,i] is the flow of special pod from k in I1 to i in I2
        # load Y to generate arriving normal pods
        self.arrival_rate_dict = {ws.id: 0. for ws in self.workstation_dict.values() if ws.type != 'I2'}
        for ji in Y.keys():
            self.arrival_rate_dict[ji[1]] += Y[ji[0],ji[1]]
        # load Z to the system
        Q_I2 = dict()
        for ki in Z.keys():
            k = ki[0]
            i = ki[1]
            if Z[k,i]> EPS:
                self.workstation_dict[k].assign_I2(self.workstation_dict[i])
            if i in Q_I2.keys():
                Q_I2[i] += Z[k,i]
            else:
                Q_I2[i] = Z[k,i]
        # stability check
        for i in self.open_I1:
            rho = self.arrival_rate_dict[i]* self.workstation_dict[i].E_service_time
            assert rho<1.-EPS
        for i in self.open_I3:
            rho = self.arrival_rate_dict[i] * self.workstation_dict[i].E_service_time
            assert rho < 1. - EPS
        for i in self.open_I2:
            if i in Q_I2.keys():
                rho = Q_I2[i] * self.workstation_dict[i].E_service_time
                assert rho < 1. - EPS

if __name__ =='__main__':
    width=20
    height=20
    min_distance = 4
    demand_density=0.05/min_distance/min_distance

    n_X = width // min_distance + 1
    n_Y = height // min_distance + 1
    X = [1. for i in range(n_Y*n_X)]
    E_S1=10.
    Var_S1=4.
    E_S2=100.
    Var_S2=16.
    special_pod_size=100
    warehouse = Warehouse(width, height, demand_density, min_distance, X, E_S1, Var_S1, E_S2, Var_S2, special_pod_size)

    simulation_steps = 100000.
    simulator = Simulator(warehouse.workstation_dict, warehouse.arrival_rate_dict, simulation_steps)
    simulator.run_simulation()
    for ws in warehouse.workstation_dict.values():
        if ws.type =='I2':
            ws.flow_stats(total_time=simulation_steps)
