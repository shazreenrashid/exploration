import numpy as np
import networkx as nx
from typing import Dict, Optional, List

class GraphEnvironment:
    def __init__(self, config, seed: Optional[int] = None):
        self.config = config 
        self.num_agents = config.get('num_agents', 4)
        self.grid_width = config.get('grid_width', 100)
        self.grid_height = config.get('grid_height', 100)
        self.sensor_range = config.get('sensor_range', 7.0)
        self.risk_threshold = config.get('risk_threshold', 0.8)
        self.noise_sd = config.get('noise_sd', 0.05)
        self.buffer_fraction= config.get('buffer_fraction', 0.1)
        self.risk_sd_min= config.get('risk_sd_min',6.0)
        self.risk_sd_max= config.get('risk_sd_max',15.0)
        if seed is not None:
            np.random.seed(seed)

        self.R_true = self._generate_risk_world()
        
        self.global_coverage = np.zeros((self.grid_width, self.grid_height), dtype=bool)
        self.agent_positions = np.zeros((self.num_agents, 2))
        self.agent_trajectories = [[] for _ in range(self.num_agents)]
        self.current_step = 0

    def _generate_risk_world(self):
        """Standard Gaussian blob generation."""
        grid = np.full((self.grid_width, self.grid_height), 0.05)
        # We can pull num_blobs from config, defaulting to 5
        num_blobs = self.config.get('num_blobs', 5)
        
        for _ in range(num_blobs):
            # Dynamic blob placement based on grid size (keeping a 10% buffer from edges)
            cx = np.random.randint(int(self.grid_width *  self.buffer_fraction), int(self.grid_width * (1-self.buffer_fraction)))
            cy = np.random.randint(int(self.grid_height * self.buffer_fraction), int(self.grid_height * (1-self.buffer_fraction)))
            sigma = np.random.uniform(self.risk_sd_min, self.risk_sd_max)
            intensity = np.random.uniform(0.7, 1.0)
            
            x, y = np.arange(0, self.grid_width), np.arange(0, self.grid_height)
            xx, yy = np.meshgrid(x, y)
            d2 = (xx - cx)**2 + (yy - cy)**2
            blob = intensity * np.exp(-d2 / (2 * sigma**2))
            grid = np.maximum(grid, blob)
        return grid

    def reset(self):
        self.current_step = 0
        self.global_coverage.fill(False)
        self.R_true = self._generate_risk_world()
        
        # DYNAMIC CORNERS: [bottom-left, bottom-right, top-left, top-right]
        corners = [
            [2, 2], 
            [self.grid_width - 3, 2], 
            [2, self.grid_height - 3], 
            [self.grid_width - 3, self.grid_height - 3]
        ]
        
        observations = {}
        for i in range(self.num_agents):
            # Safe assignment: if there are >4 agents, wrap around to start
            spawn_pos = corners[i % len(corners)]
            self.agent_positions[i] = np.array(spawn_pos, dtype=float)
            self.agent_trajectories[i] = [self.agent_positions[i].copy()]
            observations[i] = [self._get_obs(i)] 
        return observations

    def _get_custom_patch(self, pos):
        px, py = int(pos[0]), int(pos[1])
        patch = np.full((5, 5), 0.5) 
        
        # DYNAMIC BOUNDARIES
        x_min, x_max = max(0, px - 2), min(self.grid_width, px + 3)
        y_min, y_max = max(0, py - 2), min(self.grid_height, py + 3)
        
        p_x_min, p_y_min = 2 - (px - x_min), 2 - (py - y_min)
        true_data = self.R_true[x_min:x_max, y_min:y_max]
        
        noise = np.random.normal(0, self.noise_sd, true_data.shape)
        patch[p_x_min:p_x_min+(x_max-x_min), p_y_min:p_y_min+(y_max-y_min)] = np.clip(true_data + noise, 0, 1)
        return patch

    def _get_obs(self, agent_id):
        pos = self.agent_positions[agent_id]
        return {"position": pos.copy(), "risk_patch": self._get_custom_patch(pos)}
    

    def step(self, actions: Dict[int, np.ndarray]):
        """
        Moves agents and returns:
        1. obs_buffers: Perception along the whole path.
        2. team_reward: Centralized reward for CTDE training.
        """
        self.current_step += 1
        obs_buffers = {i: [] for i in range(self.num_agents)}
        
        # --- NEW: Pull weights from config (with fallbacks) ---
        w_exploration = self.config.get('w_exploration', 0.1)
        w_risk = self.config.get('w_risk', 20.0)
        w_deconfliction = self.config.get('w_deconfliction', 5.0)
        physical_deconfliction_radius = self.config.get('physical_deconfliction_radius', 10.0)
        steepness = self.config.get('steepness',5.0) 

        # Track stats for this specific team step
        new_pixels_covered = 0
        total_risk_penalty = 0.0

        for i, target_pos in actions.items():
            start_pos = self.agent_positions[i].copy()
            
            # --- FIX 1: DYNAMIC BOUNDARIES ---
            # Clip X and Y independently based on the true grid size
            target_x = np.clip(target_pos[0], 0, self.grid_width - 1)
            target_y = np.clip(target_pos[1], 0, self.grid_height - 1)
            target_pos = np.array([target_x, target_y])
            
            # 1. Path Generation (Sampling every 3 units)
            dist = np.linalg.norm(target_pos - start_pos)
            sensor_interval = 3.0 # How far the agent walks before taking a reading
            
            path_coords = []
            if dist < 1e-5:
                path_coords.append(start_pos)
            else:
                # Get the directional vector
                direction = (target_pos - start_pos) / dist
                
                # Walk along the line, dropping a point every 3.0 units
                for d in np.arange(0, dist, sensor_interval):
                    path_coords.append(start_pos + direction * d)
                
                # Always append the exact final target position to ensure we arrive
                if np.linalg.norm(path_coords[-1] - target_pos) > 1e-5:
                    path_coords.append(target_pos)

            # 2. Sequential Sensing
            for curr_p in path_coords:
                ix, iy = int(curr_p[0]), int(curr_p[1])
                
                # A. Perception Buffer
                obs_buffers[i].append({'position': curr_p, 'risk_patch': self._get_custom_patch(curr_p)})

                # B. Team Exploration Reward
                # --- FIX 2: DYNAMIC FOOTPRINT ---
                x_s, x_e = max(0, ix-2), min(self.grid_width, ix+3)
                y_s, y_e = max(0, iy-2), min(self.grid_height, iy+3)
                
                # Calculate newly explored pixels
                new_mask = ~self.global_coverage[x_s:x_e, y_s:y_e]
                new_pixels_covered += np.sum(new_mask)
                self.global_coverage[x_s:x_e, y_s:y_e] = True

                # C. Safety Penalty
                # --- FIX 3: CONFIG RISK WEIGHT ---
                risk_val = self.R_true[ix, iy]
                total_risk_penalty += risk_val * w_risk 

            self.agent_positions[i] = target_pos
            self.agent_trajectories[i].append(target_pos.copy())

        # 3. Calculate CENTRALIZED REWARD
        # Reward = (Exploration Gain) - (Team Safety Penalty) - (Distance Overlap Penalty)
        reward = (new_pixels_covered * w_exploration) - total_risk_penalty

        # Deconfliction Penalty: Punish agents for being too close to each other
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                d = np.linalg.norm(self.agent_positions[i] - self.agent_positions[j])
                
                if d < physical_deconfliction_radius:
                    # Normalizes distance from 0 to 1 inside the radius.
                    overlap = 1.0 - (d / physical_deconfliction_radius)
                    
                    # --- THE FIX: The Steepness Factor ---
                    
                    
                    # np.exp(5.0 * 1.0) = ~148. 
                    # If w_deconfliction is 5.0, a direct collision is a ~740 penalty!
                    penalty = w_deconfliction * (np.exp(steepness * overlap) - 1.0)
                    
                    reward -= penalty
        

        done = self.current_step >= self.config.get('max_steps', 200)
        return obs_buffers, reward, done, {}


    def render(self, agents=None):
        import matplotlib.pyplot as plt
        
        # Initialize the figure on the first call
        if not hasattr(self, 'fig'):
            # Creating a 2x3 layout: Top 2 for Global context, Bottom 4 for Agent beliefs
            self.fig = plt.figure(figsize=(15, 8))
            self.axes = []
            self.axes.append(self.fig.add_subplot(2, 3, 1)) # Global Risk
            self.axes.append(self.fig.add_subplot(2, 3, 2)) # Global Trajectories
            for i in range(4):
                self.axes.append(self.fig.add_subplot(2, 3, i + 3))
        
        for ax in self.axes: ax.clear()
        colors = ['blue', 'green', 'purple', 'orange']

        # 1. Global Ground Truth
        self.axes[0].imshow(self.R_true.T, cmap='hot', origin='lower', extent=[0, 100, 0, 100])
        self.axes[0].set_title("Physical World (Ground Truth)")

        # 2. Global Trajectories (The old render logic)
        self.axes[1].imshow(self.R_true.T, cmap='hot', origin='lower', alpha=0.3, extent=[0, 100, 0, 100])
        for i in range(self.num_agents):
            path = np.array(self.agent_trajectories[i])
            if len(path) > 0:
                self.axes[1].plot(path[:, 0], path[:, 1], color=colors[i], label=f'A{i}')
            self.axes[1].scatter(self.agent_positions[i,0], self.agent_positions[i,1], color=colors[i], s=40)
        self.axes[1].set_title("Global Path Tracking")

        # 3. Individual Agent Beliefs
        if agents:
            for i, agent in enumerate(agents):
                ax = self.axes[i+2]
                # Show the agent's internal R matrix
                # 0.5 = Gray/Orange (Unknown), 0.0 = White (Safe), 1.0 = Red (Danger)
                im = ax.imshow(agent.belief.R.T, cmap='YlOrRd', origin='lower', 
                               extent=[0, 100, 0, 100], vmin=0, vmax=1)
                
                # Draw their graph on top
                graph = agent.belief.graph
                for u, v in graph.edges():
                    p1, p2 = graph.nodes[u]['pos'], graph.nodes[v]['pos']
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors[i], alpha=0.2, linewidth=0.5)
                
                ax.scatter(agent.current_pos[0], agent.current_pos[1], color=colors[i], s=30)
                ax.set_title(f"Agent {i} Belief (Risk Map)")

        plt.tight_layout()
        plt.pause(0.01)