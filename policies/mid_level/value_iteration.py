import numpy as np

class RiskAwareValueIteration:
    def __init__(self, config): 
        # Pull values from config, or use defaults if they don't exist
        self.gamma = config.get('vi_gamma', 0.8) 
        self.risk_weight = config.get('risk_weight', 1500.0)
        self.ki = config.get('ki', 100.0)
        self.risk_threshold = config.get('risk_threshold', 0.6)
        
        # --- FIX 1: DYNAMIC BOUNDARIES & SENSORS ---
        self.grid_width = config.get('grid_width', 100)
        self.grid_height = config.get('grid_height', 100)
        self.sensor_patch = config.get('sensor_patch', 5)
        self.max_info_pixels = float(self.sensor_patch ** 2)
        
        # How many times value propagates (default 30)
        self.vi_iters = config.get('vi_iters', 30)
        self.edge_risk_weight= config.get('edge_risk_weight', 20)
        self.vi_nr_steepness= config.get('vi_nr_steepness', 5)

    def solve(self, subgraph, mental_c_map, target_node_list):
        """
        Cluster-Targeted VI: 
        Only nodes in the target_node_list act as primary reward sources.
        Value propagates through the graph to find the safest tactical path 
        to 'clean out' the cluster.
        """
        # 1. Initialize all nodes with a baseline low value
        V = {n: 0.0 for n in subgraph.nodes()}
        
        # 2. Assign intrinsic rewards ONLY to frontiers within the targeted cluster
        # This creates the "gravity well" that keeps the agent inside the cluster
        target_set = set(target_node_list)
        
        # --- FIX 2: CONFIGURABLE ITERATIONS ---
        for _ in range(self.vi_iters):
            V_new = V.copy()
            for u in subgraph.nodes():
                # --- A. INTRINSIC NODE REWARD ---
                pos_u = subgraph.nodes[u]['pos']
                px, py = int(pos_u[0]), int(pos_u[1])
                
                # --- FIX 3: DYNAMIC CLIPPING ---
                x_s, x_e = max(0, px-2), min(self.grid_width, px+3)
                y_s, y_e = max(0, py-2), min(self.grid_height, py+3)
                
                # Risk penalty is universal (Always avoid fire)
                risk_u = subgraph.nodes[u].get('risk', 0.5)
                node_risk_penalty = self.risk_weight * (np.exp(risk_u * self.vi_nr_steepness) - 1)
                
                # Info gain is only a 'Reward' if the node is in our target cluster
                if u in target_set:
                    unseen = np.sum(mental_c_map[x_s:x_e, y_s:y_e] == 0)
                    # --- FIX 4: DYNAMIC INFO NORMALIZATION ---
                    info_gain = (unseen / self.max_info_pixels) * self.ki
                else:
                    info_gain = 0.0
                
                intrinsic_reward = info_gain - node_risk_penalty

                # --- B. NEIGHBOR UTILITY (BELLMAN UPDATE) ---
                max_Q = -float('inf')
                neighbors = list(subgraph.neighbors(u))
                
                if not neighbors:
                    V_new[u] = intrinsic_reward
                    continue

                for v in neighbors:
                    # Path Cost logic remains identical to preserve safety
                    dist = subgraph.get_edge_data(u, v).get('weight', 1.0)
                    risk_v = subgraph.nodes[v].get('risk', 0.5)
                    avg_edge_risk = (risk_u + risk_v) / 2.0
                    
                    if avg_edge_risk > self.risk_threshold:
                        edge_cost = 1000000.0  
                    else:
                        edge_cost = dist * (1 + self.edge_risk_weight * avg_edge_risk)
                    
                    # Bellman equation: Q = local context + future potential
                    Q = intrinsic_reward + (self.gamma * V[v]) - edge_cost
                    
                    if Q > max_Q:
                        max_Q = Q
                
                V_new[u] = max_Q
            V = V_new

        # 3. Write results back to the graph nodes
        # The Agent.act() method will use these values to pick the best neighbor
        for n in subgraph.nodes():
            subgraph.nodes[n]['value'] = V.get(n, -float('inf'))



