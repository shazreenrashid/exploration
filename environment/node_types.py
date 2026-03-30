import numpy as np
from enum import Enum
import yaml

class NodeType(Enum):
    BREADCRUMB = 0
    FRONTIER = 1

class Node:
    def __init__(self, node_id, position, risk, node_type: NodeType, grid_width, grid_height, sensor_patch):
        self.id = node_id
        self.position = np.array(position, dtype=np.float32)
        self.risk = float(risk)
        self.node_type = node_type
        self.grid_width= grid_width
        self.grid_height= grid_height
        self.sensor_patch= sensor_patch
        # Metadata for GNN and Tracking
        self.timestamp = 0
        self.visit_count = 0
        self.unseen_count = 0.0  # Number of unexplored pixels in 5x5 window
        
        # Decentralized Policy Attributes
        self.cluster_id = -1     # Assigned by the Actor (Clustering Layer)
        self.claimed_by = -1     # ID of agent who broadcasted intent for this node
        self.embedding = None    # Store GAT output embedding here if needed

    def to_feature_vector(self):
        """
        Converts node data to a vector for GNN input.
        Normalized to [0, 1] range for neural network stability.
        """
        # Feature Vector: [Norm_X, Norm_Y, Risk, Type_OneHot_F, Type_OneHot_B, Utility, Claimed_Mask]
        return np.array([
            self.position[0] / self.grid_width,              # 0: Normalized X
            self.position[1] / self.grid_height,              # 1: Normalized Y
            self.risk,                             # 2: Risk [0, 1]
            1.0 if self.node_type == NodeType.FRONTIER else 0.0, # 3: Is Frontier
            1.0 if self.node_type == NodeType.BREADCRUMB else 0.0, # 4: Is Breadcrumb
            self.unseen_count / (self.sensor_patch)**2,              # 5: Normalized Info Gain Utility
            1.0 if self.claimed_by != -1 else 0.0  # 6: Broadcast Mask (External claim)
        ], dtype=np.float32)

class FrontierNode(Node):
    def __init__(self, node_id, position, risk, grid_width, grid_height, sensor_patch, unseen_count=0):
        # Pass all the layout variables up to the main Node class
        super().__init__(node_id, position, risk, NodeType.FRONTIER, grid_width, grid_height, sensor_patch)
        self.unseen_count = float(unseen_count)

class BreadcrumbNode(Node):
    def __init__(self, node_id, position, risk, grid_width, grid_height, sensor_patch, unseen_count=0):
        # Pass all the layout variables up to the main Node class
        super().__init__(node_id, position, risk, NodeType.BREADCRUMB, grid_width, grid_height, sensor_patch)
        self.visit_count = 1
        self.unseen_count = float(unseen_count)