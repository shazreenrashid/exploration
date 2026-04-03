import numpy as np
import torch
from environment.node_types import NodeType
from torch.distributions import Categorical

class Agent:
    def __init__(self, agent_id, config):
        self.agent_id = agent_id
        self.config = config
        self.current_node_id = None 
        
        # Core Parameters
        self.grid_w = config.get('grid_width', 100)
        self.grid_h = config.get('grid_height', 100)
        self.sensor_range = config.get('sensor_range', 7.0)
        self.num_clusters = config.get('num_clusters', 4)
        
        # State
        self.current_pos = np.zeros(2)
        self.target_cluster_centroid = None  # For broadcasting to peers
        
        # --- NEW: Replan Triggers ---
        self.replan_interval = config.get('replan_interval', 20) # Replan every X steps
        self.steps_since_replan = 0
        
        # 1. The Belief Model (Topological + Metric)
        from agents.belief_model import BeliefModel
        self.belief = BeliefModel(
            agent_id=agent_id, 
            grid_width=self.grid_w,
            grid_height=self.grid_h,
            sensor_patch=config.get('sensor_patch', 5),
            d_min=config.get('d_min', 2.5), 
            d_max=config.get('d_max', 8.0),
            deconfliction_radius=config.get('deconfliction_radius', 12.0)
        )
        
        # 2. High-Level Brain: GAT Actor-Critic
        from policies.high_level.gat_actor_critic import GATActorCritic
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = GATActorCritic(
            input_dim=7,
            hidden_dim=config.get('hidden_dim', 64),
            num_clusters=self.num_clusters
        ).to(self.device)

        self.last_embedding = None
        self.cached_assignments = None       # frozen K-Means assignments between replans
        self.cached_target_nodes = None      # frozen target frontier list
        self.cached_kmeans_centroids = None  # cluster centres for assigning new nodes
        self.current_cluster_id = None       # which cluster this agent currently holds
        self.needs_replan = True
                
        # 3. Mid-Level Brain: Value Iteration
        from policies.mid_level.value_iteration import RiskAwareValueIteration
        self.vi_solver = RiskAwareValueIteration(config)

    def update_perception(self, obs_buffer, step):
        """Processes buffer and stamps persistent risk."""
        for obs in obs_buffer:
            pos, patch = obs['position'], obs['risk_patch']
            ix, iy = int(pos[0]), int(pos[1])
            
            # Global Map Boundaries
            x_s, x_e = max(0, ix-2), min(self.grid_w, ix+3)
            y_s, y_e = max(0, iy-2), min(self.grid_h, iy+3)
            
            # Calculate where the valid data starts inside the 5x5 patch (Boundary Fix)
            p_x_start = 2 - (ix - x_s)
            p_y_start = 2 - (iy - y_s)
            p_x_end = p_x_start + (x_e - x_s)
            p_y_end = p_y_start + (y_e - y_s)
            
            # Extract the physically aligned data
            patch_data = patch[p_x_start:p_x_end, p_y_start:p_y_end]
            
            current_belief = self.belief.R[x_s:x_e, y_s:y_e]
            coverage = self.belief.C[x_s:x_e, y_s:y_e]
            
            new_values = np.where(coverage == 0, patch_data, 0.8 * current_belief + 0.2 * patch_data)
            self.belief.R[x_s:x_e, y_s:y_e] = new_values
            self.belief.C[x_s:x_e, y_s:y_e] = 1 

        final_obs = obs_buffer[-1]
        self.current_pos = final_obs['position']
        self.current_node_id = self.belief.add_or_update_node(
            pos=self.current_pos,
            risk=final_obs['risk_patch'][2, 2],
            node_type=NodeType.BREADCRUMB,
            step=step
        )
        self.generate_frontiers(step)

    def generate_frontiers(self, step):
        """Standard boundary-detection frontier generation."""
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        for angle in angles:
            full_dist = self.sensor_range * 0.9
            for d in np.arange(full_dist, 0, -1):
                fx = int(self.current_pos[0] + np.cos(angle) * d)
                fy = int(self.current_pos[1] + np.sin(angle) * d)
                
                # Dynamic Clipping
                fx = np.clip(fx, 0, self.grid_w - 1)
                fy = np.clip(fy, 0, self.grid_h - 1)
                
                if self.belief.C[fx, fy] == 1:
                    self.belief.add_or_update_node(
                        pos=np.array([fx, fy]),
                        risk=self.belief.R[fx, fy],
                        node_type=NodeType.FRONTIER,
                        step=step
                    )
                    break

    def get_broadcast_summary(self):
        """Returns the centroid of the claimed cluster for multi-agent deconfliction."""
        if self.target_cluster_centroid is None:
            return None
        return {'centroid': self.target_cluster_centroid, 'agent_id': self.agent_id}
    
    def act(self, step, train=False, claimed_cluster_ids=None):
        """
        Hierarchical Action Phase with cluster commitment:
        - Replan (run GAT + KMeans) on interval or when cluster is exhausted.
        - Between replans: run VI strictly on cached target nodes.
        - Sequential greedy: avoids clusters claimed by higher-priority agents,
          skips empty clusters, and falls back to nearest overall frontier if needed.
        """
        if claimed_cluster_ids is None:
            claimed_cluster_ids = set()

        if self.belief.graph.number_of_nodes() < 2:
            self.last_embedding = torch.zeros(
                1, self.policy.hidden_dim * self.num_clusters, device=self.device
            )
            if train:
                return self.current_pos, torch.tensor(0.0, device=self.device)
            return self.current_pos

        pyg_data = self.belief.get_pyg_data()

        # --- TRIGGER 1: Step Timeout ---
        if self.steps_since_replan >= self.replan_interval:
            self.needs_replan = True

        # ----------------------------------------------------------------
        # REPLAN PATH: run full GAT + KMeans
        # ----------------------------------------------------------------
        if self.needs_replan:
            cluster_probs, _, z_flat, assignments, centroids = self.policy(
                pyg_data, cached_assignments=None
            )
            self.last_embedding = z_flat.detach()
            self.cached_assignments = assignments
            self.cached_kmeans_centroids = centroids
            
            # Reset the step counter upon a fresh replan
            self.steps_since_replan = 0 
            self.needs_replan = False

            node_ids = list(self.belief.graph.nodes)
            cluster_order = torch.argsort(
                cluster_probs.squeeze(0), descending=True
            ).tolist()

            target_nodes = []
            chosen_cluster_id = None
            
            # --- THE NEW FALLBACK LOGIC ---
            # Loop through the clusters from highest probability to lowest
            for candidate_cluster in cluster_order:
                # if candidate_cluster in claimed_cluster_ids:
                #     continue
                    
                candidate_nodes = [
                    node_ids[k] for k, c_id in enumerate(assignments)
                    if c_id == candidate_cluster and
                    self.belief.graph.nodes[node_ids[k]]['obj'].node_type == NodeType.FRONTIER
                ]
                
                if candidate_nodes:
                    # Success! We found the highest-ranked cluster that actually has frontiers.
                    target_nodes = candidate_nodes
                    chosen_cluster_id = candidate_cluster
                    break # Stop looking, we found our target!
                else:
                    # This cluster is empty. Let the loop naturally continue to the next best cluster.
                    continue

            # Calculate log probabilities for training based on the network's output
            if train:
                m = Categorical(cluster_probs.squeeze(0))
                if chosen_cluster_id is not None:
                    log_prob = m.log_prob(
                        torch.tensor(chosen_cluster_id, device=self.device)
                    )
                else:
                    log_prob = torch.tensor(0.0, device=self.device)
            else:
                log_prob = None

            # Global Fallback: If ALL clusters were empty or claimed
            if not target_nodes:
                frontiers = self.belief.get_frontier_nodes()
                if not frontiers:
                    # Map is 100% explored, nothing left to do
                    if train: return self.current_pos, log_prob
                    return self.current_pos
                
                # Pick the closest global frontier as a last resort
                target_nodes = [min(
                    frontiers,
                    key=lambda n: np.linalg.norm(
                        self.current_pos - self.belief.graph.nodes[n]['pos']
                    )
                )]

            self.cached_target_nodes = target_nodes
            self.current_cluster_id = chosen_cluster_id

        # ----------------------------------------------------------------
        # CACHED PATH: Reuse exact target list, no new assignments
        # ----------------------------------------------------------------
        else:
            cluster_probs, _,  z_flat, assignments, _ = self.policy(
                pyg_data, cached_assignments=self.cached_assignments
            )
            self.last_embedding = z_flat.detach()

            # Filter the cached list to only nodes that are STILL frontiers
            valid_targets = [
                n for n in self.cached_target_nodes
                if n in self.belief.graph.nodes and
                self.belief.graph.nodes[n]['obj'].node_type == NodeType.FRONTIER
            ]

            # --- TRIGGER 2: Cluster Exhausted ---
            if not valid_targets:
                self.needs_replan = True
                # Instantly recurse to trigger the replan path on this exact step
                return self.act(step, train=train, claimed_cluster_ids=claimed_cluster_ids)

            self.cached_target_nodes = valid_targets

            if train:
                m = Categorical(cluster_probs.squeeze(0))
                log_prob = m.log_prob(
                    torch.tensor(self.current_cluster_id, device=self.device)
                ) if self.current_cluster_id is not None else torch.tensor(0.0, device=self.device)
            else:
                log_prob = None

            target_nodes = self.cached_target_nodes

        # ----------------------------------------------------------------
        # SHARED: update centroid, run VI, take tactical step
        # ----------------------------------------------------------------
        target_positions = np.array([
            self.belief.graph.nodes[n]['pos'] for n in target_nodes
        ])
        self.target_cluster_centroid = np.mean(target_positions, axis=0)

        self.vi_solver.solve(
            subgraph=self.belief.graph,
            mental_c_map=self.belief.C,
            target_node_list=target_nodes
        )

        neighbors = [
            n for n in self.belief.graph.neighbors(self.current_node_id)
            if n != self.current_node_id
        ]

        final_pos = self.current_pos
        if neighbors:
            best_neighbor = max(
                neighbors,
                key=lambda n: self.belief.graph.nodes[n].get('value', -1e9)
            )
            final_pos = self.belief.graph.nodes[best_neighbor]['pos']

        # Increment step counter for timeout tracking
        self.steps_since_replan += 1

        if train:
            return final_pos, log_prob
        return final_pos

    def reset(self):
        from agents.belief_model import BeliefModel
        self.belief = BeliefModel(
            agent_id=self.agent_id,
            grid_width=self.grid_w,
            grid_height=self.grid_h,
            sensor_patch=self.config.get('sensor_patch', 5),
            d_min=self.config.get('d_min', 2.5),
            d_max=self.config.get('d_max', 8.0),
            deconfliction_radius=self.config.get('deconfliction_radius', 12.0)
        )
        self.current_pos = np.zeros(2)
        self.current_node_id = None
        self.target_cluster_centroid = None
        self.last_embedding = None
        self.cached_assignments = None
        self.cached_target_nodes = None
        self.cached_kmeans_centroids = None
        self.current_cluster_id = None
        self.needs_replan = True
        self.steps_since_replan = 0


