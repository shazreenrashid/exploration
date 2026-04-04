import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.spatial import KDTree
from environment.node_types import NodeType, Node, FrontierNode, BreadcrumbNode

class BeliefModel:
    def __init__(self, agent_id, grid_height=100, grid_width=100, sensor_patch=5, d_min=2.5, d_max=8.0, deconfliction_radius=12):
        self.agent_id = agent_id
        self.grid_width = grid_width
        self.grid_height = grid_height     
        self.sensor_patch = sensor_patch
        self.d_min = d_min
        self.d_max = d_max
        self.deconfliction_radius= deconfliction_radius
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 1. Metric Belief (Matrices) ---
        # C: Coverage matrix (0: unseen, 1: seen)
        self.C = np.zeros((grid_width, grid_height), dtype=np.float32)
        # R: Risk matrix (0.5 default/unknown)
        self.R = np.full((grid_width, grid_height), 0.5, dtype=np.float32)
        
        # --- 2. Topological Belief (Graph) ---
        self.graph = nx.Graph()
    
    def add_or_update_node(self, pos, risk, node_type: NodeType, step):
        """
        Adds or updates nodes using the specialized Node classes.
        Calculates local Info Gain (unseen_count) for the GAT features.
        """
        existing_node = self._find_nearest_node(pos)
        
        # Calculate local info gain (utility) centered on the NEW position 
        # (We only use this if we end up creating a brand new node)
        px, py = int(pos[0]), int(pos[1])
        x_s, x_e = max(0, px-2), min(self.grid_width, px+3)
        y_s, y_e = max(0, py-2), min(self.grid_height, py+3)
        new_unseen_count = np.sum(self.C[x_s:x_e, y_s:y_e] == 0)

        if existing_node:
            node_id, dist = existing_node
            if dist < self.d_min:
                node_obj = self.graph.nodes[node_id]['obj']
                
                # Calculate updates based on the EXISTING node's exact position ---
                old_pos = node_obj.position
                opx, opy = int(old_pos[0]), int(old_pos[1])
                
                # 1. Get the up-to-date risk exactly at the node's location
                true_node_risk = self.R[opx, opy]
                
                # 2. Calculate the unseen count exactly centered on the node's location
                ox_s, ox_e = max(0, opx-2), min(self.grid_width, opx+3)
                oy_s, oy_e = max(0, opy-2), min(self.grid_height, opy+3)
                true_unseen_count = np.sum(self.C[ox_s:ox_e, oy_s:oy_e] == 0)

                # Now apply these spatially accurate values
                node_obj.risk = float(true_node_risk)
                node_obj.unseen_count = float(true_unseen_count)
                self.graph.nodes[node_id]['risk'] = float(true_node_risk) 
                
                # Conversion logic: Frontier -> Breadcrumb
                if node_obj.node_type == NodeType.FRONTIER and node_type == NodeType.BREADCRUMB:
                    node_obj.node_type = NodeType.BREADCRUMB
                    self.graph.nodes[node_id]['type'] = NodeType.BREADCRUMB 
                    node_obj.timestamp = step
                
                return node_id

        # --- Creating a New Node (Uses the 'pos' arguments) ---
        new_id = f"a{self.agent_id}_n{len(self.graph.nodes)}"
        
        if node_type == NodeType.FRONTIER:
            node_obj = FrontierNode(new_id, pos, risk, self.grid_width, self.grid_height, self.sensor_patch, new_unseen_count)
        else:
            node_obj = BreadcrumbNode(new_id, pos, risk, self.grid_width, self.grid_height, self.sensor_patch, new_unseen_count)
            node_obj.timestamp = step

        self.graph.add_node(
            new_id,
            pos=np.array(pos),
            obj=node_obj, 
            type=node_type, 
            risk=float(risk))
        
        self._update_edges(new_id)
        return new_id

    def get_pyg_data(self):
        """
        Converts the NetworkX graph into a PyTorch Geometric Data object for GAT.
        Edge attributes encode a smooth risk-aware traversal cost that mirrors
        the VI solver's path cost formula without any hard thresholding.
        """
        node_ids = list(self.graph.nodes)
        node_map = {node_id: i for i, node_id in enumerate(node_ids)}

        # 1. Generate Feature Matrix (X) [N, 7]
        x_features = [self.graph.nodes[n_id]['obj'].to_feature_vector() for n_id in node_ids]
        x = torch.tensor(np.array(x_features), dtype=torch.float)

        # 2. Generate Edge Index (COO format) and Edge Attributes
        edge_list = []
        edge_attr = []
        max_dim = max(self.grid_width, self.grid_height)

        for u, v, edge_data in self.graph.edges(data=True):
            edge_list.append([node_map[u], node_map[v]])
            edge_list.append([node_map[v], node_map[u]])

            dist = edge_data.get('weight', 1.0)
            risk_u = self.graph.nodes[u].get('risk', 0.5)
            risk_v = self.graph.nodes[v].get('risk', 0.5)
            avg_edge_risk = (risk_u + risk_v) / 2.0

            # Smooth risk-aware cost — same formula as VI, no hard wall.
            # High-risk edges naturally get much larger costs due to the
            # multiplicative risk term, without any discontinuity.
            cost_norm = dist * (1.0 + 20.0 * avg_edge_risk) / max_dim

            edge_attr.append([cost_norm])
            edge_attr.append([cost_norm])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float)

        # 3. Spatial positions for clustering and attention
        pos = torch.tensor(
            np.array([self.graph.nodes[n]['pos'] for n in node_ids]), dtype=torch.float
        )

        pyg_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

        if hasattr(self, 'device'):
            return pyg_data.to(self.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            return pyg_data.to(device)

    def reset_broadcast_masks(self):
        """Clears all peer claims before a new coordination cycle."""
        for n_id in self.graph.nodes:
            self.graph.nodes[n_id]['obj'].claimed_by = -1

    def update_broadcast_masks(self, peer_summaries):
        """
        Marks nodes as 'claimed' if they fall within the radius of a peer's target.
        """
        for summary in peer_summaries:
            if summary is None: continue
            centroid = summary['centroid']
            peer_id = summary['agent_id']
            
            for n_id in self.graph.nodes:
                node_obj = self.graph.nodes[n_id]['obj']
                dist = np.linalg.norm(node_obj.position - centroid)
                
                # If a peer is targeting this area, mask it for our GAT attention
                if dist < self.deconfliction_radius: # Radius for deconfliction
                    node_obj.claimed_by = peer_id

    def _find_nearest_node(self, pos):
        if not self.graph.nodes: return None
        node_ids = list(self.graph.nodes)
        node_pos = np.array([self.graph.nodes[n]['pos'] for n in node_ids])
        tree = KDTree(node_pos)
        dist, idx = tree.query(pos)
        return node_ids[idx], dist

    def _update_edges(self, new_id):
        new_pos = self.graph.nodes[new_id]['pos']
        for other_id, data in self.graph.nodes(data=True):
            if other_id == new_id: continue
            dist = np.linalg.norm(new_pos - data['pos'])
            if dist <= self.d_max:
                # weight is used for VI pathfinding, distance is for GAT
                self.graph.add_edge(new_id, other_id, weight=dist)

    def get_frontier_nodes(self):
        """Helper to get all current frontiers."""
        return [n for n, d in self.graph.nodes(data=True) if d['type'] == NodeType.FRONTIER]

