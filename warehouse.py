import time

from workstations import Workstation_I1, Workstation_I2, Workstation_I3
from util import EPS, plot_colored_grid, workstation_travel_time
from simulator import Simulator
import numpy as np
import gurobipy as gp
from gurobipy import GRB

env = gp.Env(empty=True)
env.setParam('OutputFlag', 0)
env.start()

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
        key_to_remove = [key for key in self.workstation_dict.keys() if X[key]<EPS]
        for key in key_to_remove:
            self.workstation_dict.pop(key, None)
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

    def plot_system(self):
        system_map = np.zeros((self.width+1, self.height+1))
        type_dict = {'I1':1, 'I2':2, 'I3':3}
        for ws in self.workstation_dict.values():
            system_map[int(ws.x), int(ws.y)] = type_dict[ws.type]
        plot_colored_grid(system_map, colors=['white','green','blue','red'],bounds=[-0.5,0.5,1.5,2.5,3.5])

    def load_YZ(self, Y: dict, Z: dict):
        # Y[j,i] is the flow from cell j to i
        # Z[k,i] is the flow of special pod from k in I1 to i in I2
        # load Y to generate arriving normal pods
        self.arrival_rate_dict = {ws.id: 0. for ws in self.workstation_dict.values() if ws.type != 'I2'}
        for ji in Y.keys():
            if Y[ji[0],ji[1]]>EPS:
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

    def solve_L1_subproblem(self, pi, alpha_dict, setting='new'):
        set_I = [ws.id for ws in self.workstation_dict.values()]
        set_I12 = [ws.id for ws in self.workstation_dict.values() if ws.type != 'I3']
        mu = [1. / self.workstation_dict[i].E_service_time for i in set_I]
        # note: due to our constraints on min number of open facility, we choose X with min alpha_dict[i]-pi[i]*mu[i]
        X_result = {i:0 for i in set_I}
        for i in set_I:
            if alpha_dict[i]-pi[i]*mu[i]<0:
                X_result[i] = 1
        if setting=='kiva':
            for i in set_I12:
                X_result[i] = 0
        L1_value = np.sum([(alpha_dict[i]-pi[i]*mu[i])*X_result[i] for i in set_I])
        return L1_value, X_result

    def solve_socp(self, alpha_dict, setting='new', is_solving_L2=False, pi=None, X_fix_value=None):
        # setting in ['new', 'kiva', 'fix_X']
        # 'new' is our setting with internal ws
        # 'kiva' is old setting

        # if is_solving_L2, we solve the L2 subproblem instead of the original SOCP
        # in L2 subproblem, all X_i=1 and alpha=0

        # if setting='fix_X', X values should be provided in X_fix_value as a dict

        set_I = [ws.id for ws in self.workstation_dict.values()]
        set_I1 = [ws.id for ws in self.workstation_dict.values() if ws.type=='I1']
        set_I2 = [ws.id for ws in self.workstation_dict.values() if ws.type=='I2']
        set_I3 = [ws.id for ws in self.workstation_dict.values() if ws.type=='I3']
        set_I13 = set_I1 + set_I3
        set_J = list(self.J_xy_dict.keys())
        m = gp.Model("warehouse_workstation", env=env)
        # add vars
        X = m.addVars(set_I, lb=0, ub=1, vtype=GRB.BINARY, name='X')
        max_demand_one_pod = np.max(self.demand_map)
        Y = m.addVars(set_J, set_I, lb=0., ub=max_demand_one_pod, vtype=GRB.CONTINUOUS, name='Y')
        Z = m.addVars(set_I1, set_I2, lb=0., ub=1./self.E_S1, vtype=GRB.CONTINUOUS, name='Z')
        Q = m.addVars(set_I, lb=0., ub=max(1./self.E_S1, 1/self.E_S2), vtype=GRB.CONTINUOUS, name='Z')
        V = m.addVars(set_I1, set_I2, lb=0, ub=1, vtype=GRB.BINARY, name='V')
        G = m.addVars(set_I, lb=0., vtype=GRB.CONTINUOUS, name='G')
        H = m.addVars(set_I, lb=0., vtype=GRB.CONTINUOUS, name='H')
        # set obj function
        # get parameters
        big_M = 1./self.E_S2
        demand_j = [self.demand_map[int(self.J_xy_dict[j][0])][int(self.J_xy_dict[j][1])] for j in set_J]
        distance_ji = dict()
        distance_ki = dict()
        mu = [1./self.workstation_dict[i].E_service_time for i in set_I]
        CV_S1_2 = self.Var_S1/self.E_S1/self.E_S1
        CV_S2_2 = self.Var_S2/self.E_S2/self.E_S2
        for j in set_J:
            for i in set_I:
                distance_ji[j,i] = abs(self.workstation_dict[i].x-self.J_xy_dict[j][0]) + \
                                   abs(self.workstation_dict[i].y-self.J_xy_dict[j][1])
        for k in set_I1:
            for i in set_I2:
                distance_ki[k,i] = workstation_travel_time(self.workstation_dict[k], self.workstation_dict[i])
        if not is_solving_L2:
            pi = {i: 0. for i in set_I} # if not solving subproblem l2, set pi to be zeros
        else:
            for i in alpha_dict.keys():
                alpha_dict[i] = 0. # if is solving l2 subproblem, set alpha=0 to remove X in the obj function

        # calculate obj function
        cost_I13_sum = gp.quicksum([
            alpha_dict[i]*X[i] + Q[i]*(1./mu[i]+pi[i])+ (1.+CV_S1_2)/2./mu[i]*G[i] +
            gp.quicksum([
                distance_ji[j,i]*Y[j,i]
                for j in set_J
            ])
            for i in set_I13
        ]) # note: pi[i]=0 if not solving l2 subproblem
        cost_I2_sum = gp.quicksum([
            alpha_dict[i] * X[i] + Q[i]* (1./ mu[i]+pi[i]) + CV_S2_2 / 2. / mu[i] * G[i] +
            gp.quicksum([
                distance_ki[k, i] * Z[k, i]
                for k in set_I1
            ])
            for i in set_I2
        ])
        m.setObjective(cost_I13_sum+cost_I2_sum, sense=GRB.MINIMIZE)
        # constraints
        m.addConstrs((Q[i] == gp.quicksum([Y[j,i] for j in set_J]) for i in set_I13), name='5_1') # 5.1
        m.addConstrs((Q[i] == gp.quicksum([Z[k,i] for k in set_I1]) for i in set_I2), name='5_2') # 5.2
        m.addConstrs((gp.quicksum([Y[j,i] for i in set_I13]) == demand_j[j] for j in set_J),name='5_10') # 5.10
        m.addConstrs((Q[i] <= X[i]*mu[i] for i in set_I), name='5_11') # 5.11
        m.addConstrs((gp.quicksum([Z[k,i] for i in set_I2])==Q[k]/self.special_pod_size for k in set_I1), name='5_12') # 5.12
        m.addConstrs((Z[k,i]<=big_M*V[k,i] for k in set_I1 for i in set_I2), name='5.13') # 5.13
        m.addConstrs((gp.quicksum([V[k,i] for i in set_I2])==1 for k in set_I1), name='5_14') # 5.14
        m.addConstrs((Q[i]*Q[i] <= G[i]*H[i] for i in set_I),name='5_23') # 5.23
        m.addConstrs((H[i] == mu[i]*X[i]-Q[i] for i in set_I), name='5_24') # 5.24
        # make sure if one I1 open, then there must be at least one I2 open
        if setting=='new':
            m.addConstr(gp.quicksum([X[i] for i in set_I1])>=1, name='at_least_one_I1')
            m.addConstr(gp.quicksum([X[i] for i in set_I2])>=1, name='at_least_one_I2')
            if is_solving_L2:
                m.addConstrs((X[i]==1 for i in set_I), name='L2_open_all')
        elif setting=='kiva':
            m.addConstr(gp.quicksum([X[i] for i in set_I1]) == 0, name='no_I1')
            m.addConstr(gp.quicksum([X[i] for i in set_I2]) == 0, name='no_I2')
            if is_solving_L2:
                m.addConstrs((X[i]==1 for i in set_I3), name='L2_open_all')
        elif setting=='fix_X':
            m.addConstrs((X[i] == X_fix_value[i] for i in set_I), name='fix_X_value')
            assert not is_solving_L2 # fix_X cannot be used to solve L2
        # solve the problem
        m.optimize()
        #print('Total cost:', m.ObjVal)
        open_cost = 0.
        #print("SOLUTION:")
        for i in set_I:
            if X[i].X > 0.99:
                # print(f"workstation {i} open")
                # print('type:', self.workstation_dict[i].type)
                open_cost += alpha_dict[i]
        #print('cost without open cost:', m.ObjVal - open_cost)
        X_values = {i: X[i].X for i in set_I}
        Y_values = {(j,i):Y[j,i].X for j in set_J for i in set_I}
        Z_values = {(k,i):Z[k,i].X for k in set_I1 for i in set_I2}
        Q_values = {i: Q[i].X for i in set_I}
        if is_solving_L2:
            L2_value = m.ObjVal
            return L2_value, Q_values
        return X_values, Y_values, Z_values, m.ObjVal

    def solve_LR(self, alpha_dict, setting='new'):
        LB = -np.inf
        UB = np.inf
        set_I = [ws.id for ws in self.workstation_dict.values()]
        mu = [1. / self.workstation_dict[i].E_service_time for i in set_I]
        pi = {i:10000. for i in set_I}
        X_with_best_UB = {i:1 for i in set_I}
        Y_with_best_UB = None
        Z_with_best_UB = None
        time0 = time.time()
        for step in range(1000):
            #print('sloving L1','---'*30)
            L1_value, X_result = self.solve_L1_subproblem(pi=pi.copy(), alpha_dict=alpha_dict.copy(), setting=setting)
            #print('sloving L2', '---' * 30)
            L2_value, Q_result = self.solve_socp(alpha_dict=alpha_dict.copy(), setting=setting, is_solving_L2=True, pi=pi.copy())
            L_pi = L1_value +L2_value
            LB = max(LB, L_pi)
            #print('sloving UB', '---' * 30)
            print('X_lb', X_result)
            try:
                X_best, Y_best, Z_best, UB_this = self.solve_socp(alpha_dict=alpha_dict.copy(), setting='fix_X', is_solving_L2=False, X_fix_value=X_result.copy())
            except:
                UB_this = np.inf

            if UB_this< UB:
                X_with_best_UB = X_best.copy()
                Y_with_best_UB = Y_best.copy()
                Z_with_best_UB = Z_best.copy()
                UB = UB_this
            subgrad = {i:(Q_result[i]-X_result[i]*mu[i]) for i in set_I}
            subgrad_length_square = np.sum(np.array(list(subgrad.values()))**2)
            for i in set_I:
                pi[i] = max(0., pi[i]+(UB-L_pi)/subgrad_length_square*subgrad[i])
            print('step',step,'best UB:',UB,'LB',LB,'<<<'*30)
            print('X_ub', X_with_best_UB)
            print({i: self.workstation_dict[i].type for i in set_I})
            print('pi', pi)
            assert UB>=LB
            if (UB-LB)<1e-3:
                break
            if time.time()-time0>600.:
                break

        return X_with_best_UB, Y_with_best_UB, Z_with_best_UB, UB

    def validate_design_using_simulation(self, X, Y, Z):
        # X,Y,Z are dict
        simulation_steps = 100000.
        self.load_X(X)
        self.load_YZ(Y, Z)
        simulator = Simulator(self.workstation_dict, self.arrival_rate_dict, simulation_steps)
        robot_number = simulator.run_simulation() # note this number doesn't include robots from pods to stations
        for ji in Y.keys():
            if Y[ji]>EPS:
                x,y = self.J_xy_dict[ji[0]]
                ws = self.workstation_dict[ji[1]]
                robot_number += Y[ji]*(abs(x-ws.x)+abs(y-ws.y)) # add number of robots from pods to stations
        print('number of robots', robot_number)
        return robot_number

if __name__ =='__main__':

    width=20
    height=20
    min_distance = 4
    demand_density=0.05/min_distance/min_distance/5

    n_X = width // min_distance + 1
    n_Y = height // min_distance + 1
    X = [1. for i in range(n_Y*n_X)]
    E_S1=10.
    Var_S1=4.
    E_S2=100.
    Var_S2=16.
    special_pod_size=100

    warehouse1 = Warehouse(width, height, demand_density, min_distance, X, E_S1, Var_S1, E_S2, Var_S2, special_pod_size)
    set_I = [ws.id for ws in warehouse1.workstation_dict.values()]
    X_with_best_UB, Y_with_best_UB, Z_with_best_UB, UB = warehouse1.solve_LR(alpha_dict={i:10. for i in set_I}, setting='new')
    robot_number1 = warehouse1.validate_design_using_simulation(X_with_best_UB,Y_with_best_UB, Z_with_best_UB)

    warehouse2 = Warehouse(width, height, demand_density, min_distance, X, E_S1, Var_S1, E_S2, Var_S2, special_pod_size)
    set_I = [ws.id for ws in warehouse1.workstation_dict.values()]
    X_with_best_UB2, Y_with_best_UB2, Z_with_best_UB2, UB2 = warehouse2.solve_LR(alpha_dict={i: 10. for i in set_I},
                                                                             setting='kiva')
    robot_number2 = warehouse2.validate_design_using_simulation(X_with_best_UB2, Y_with_best_UB2, Z_with_best_UB2)

    print(robot_number1, robot_number2)
    # print('new','=='*50)
    # x1, y1, z1 = warehouse1.solve_socp(alpha_dict={i:10. for i in set_I}, setting='new')
    # simulation_steps = 100000.
    # warehouse1.load_X(x1)
    # warehouse1.load_YZ(y1, z1)
    # simulator1 = Simulator(warehouse1.workstation_dict, warehouse1.arrival_rate_dict, simulation_steps)
    # robot_number1 = simulator1.run_simulation()
    # print('number of robots 1', robot_number1)
    #
    # print('kiva', '==' * 50)
    # warehouse2 = Warehouse(width, height, demand_density, min_distance, X, E_S1, Var_S1, E_S2, Var_S2, special_pod_size)
    # x2, y2, z2 = warehouse2.solve_socp(alpha_dict={i:10. for i in set_I}, setting='kiva')
    # simulation_steps = 100000.
    # warehouse2.load_X(x2)
    # warehouse2.load_YZ(y2, z2)
    # simulator2 = Simulator(warehouse2.workstation_dict, warehouse2.arrival_rate_dict, simulation_steps)
    # robot_number2 = simulator2.run_simulation()
    # print('number of robots 2', robot_number2)

    #warehouse.plot_system()
    # simulation_steps = 100000.
    # simulator = Simulator(warehouse.workstation_dict, warehouse.arrival_rate_dict, simulation_steps)
    # simulator.run_simulation()
    # for ws in warehouse.workstation_dict.values():
    #     if ws.type =='I2':
    #         ws.flow_stats(total_time=simulation_steps)
