import casadi as ca
import numpy as np

class MPCController:
    def __init__(self, config):
        self.N = 10 
        self.dt = 0.2
        self.risk_threshold = config.get('risk_threshold', 0.7)
        self.grid_w = config.get('grid_width', 100)
        self.grid_h = config.get('grid_height', 100)

        # Create the Interpolant once to save memory
        # This turns the 100x100 grid into a continuous function
        x_grid = np.linspace(0, self.grid_w - 1, self.grid_w)
        y_grid = np.linspace(0, self.grid_h - 1, self.grid_h)
        self.risk_interp = ca.interpolant('risk_lookup', 'linear', [x_grid, y_grid])
        
    def compute_action(self, current_pos, target_pos, risk_map):
        self.opti = ca.Opti()
        x = self.opti.variable(2, self.N + 1)
        u = self.opti.variable(2, self.N)

        # 1. Base Costs
        dist_cost = ca.sumsqr(x[:, -1] - target_pos)
        ctrl_cost = 0.1 * ca.sumsqr(u)
        
        # 2. NEW: Continuous Risk Cost
        # We sample risk at every step of the predicted horizon
        total_risk_cost = 0
        weight_risk = 50.0 # High weight makes agents very "scared"
        
        for k in range(self.N + 1):
            # Query the interpolant for the risk at the predicted (x, y)
            risk_val = self.risk_interp([ca.vertcat(x[0, k], x[1, k])])
            total_risk_cost += weight_risk * risk_val

        self.opti.minimize(dist_cost + ctrl_cost + total_risk_cost)

        # 3. Standard Constraints (Dynamics, Velocity, Boundaries)
        for k in range(self.N):
            self.opti.subject_to(x[:, k+1] == x[:, k] + u[:, k] * self.dt)
            self.opti.subject_to(self.opti.bounded(-5.0, u[:, k], 5.0))
            self.opti.subject_to(self.opti.bounded(0, x[0, k+1], self.grid_w - 1))
            self.opti.subject_to(self.opti.bounded(0, x[1, k+1], self.grid_h - 1))

        self.opti.subject_to(x[:, 0] == current_pos)
        
        # Solver setup
        self.opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
        
        try:
            sol = self.opti.solve()
            path = sol.value(x)
            return path, True
        except:
            return None, False