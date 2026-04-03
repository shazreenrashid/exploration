import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
import numpy as np



class GATActorCritic(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_clusters=4):
        super().__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim

        # 1. Shared GAT Backbone
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True, edge_dim=1)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, edge_dim=1)

        # 2. Actor Head
        self.actor_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 3. ✅ NEW: Local Critic Head
        # Evaluates the pooled state for THIS agent only.
        self.critic_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_clusters, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, cached_assignments=None):
        device = data.x.device
        x, edge_index = data.x, data.edge_index

        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Step 1: GAT Encoder
        h = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        h = self.conv2(h, edge_index, edge_attr=edge_attr)  

        # Step 2: Clustering
        node_positions = data.pos.detach().cpu().numpy()

        if cached_assignments is None:
            actual_clusters = min(len(node_positions), self.num_clusters)
            kmeans = KMeans(n_clusters=actual_clusters, n_init=5, random_state=None)
            assignments = kmeans.fit_predict(node_positions)
            centroids = kmeans.cluster_centers_
        else:
            n_current = len(node_positions)
            n_cached = len(cached_assignments)

            if n_current > n_cached:
                padding = np.full(n_current - n_cached, -1, dtype=cached_assignments.dtype)
                assignments = np.concatenate([cached_assignments, padding])
            else:
                assignments = cached_assignments

            valid = assignments[assignments >= 0]
            actual_clusters = int(np.max(valid)) + 1 if len(valid) > 0 else 0
            centroids = None

        # Step 3: Pool embeddings PER CLUSTER
        assignments_tensor = torch.tensor(assignments, dtype=torch.long, device=device)
        cluster_embeddings = []
        claimed_fractions = []  

        for c in range(actual_clusters):
            mask = (assignments_tensor == c)  
            if mask.any():
                cluster_embeddings.append(h[mask].mean(dim=0))
                claimed_fractions.append(x[mask, 6].mean().unsqueeze(0))
            else:
                cluster_embeddings.append(torch.zeros(self.hidden_dim, device=device))
                claimed_fractions.append(torch.zeros(1, device=device))

        # Step 4: Handle empty graph edge case
        if actual_clusters == 0:
            z_clusters = torch.zeros((0, self.hidden_dim), device=device)
            logits = torch.zeros((0,), device=device)
        else:
            z_clusters = torch.stack(cluster_embeddings)          
            cf_tensor = torch.stack(claimed_fractions)            

            z_with_cf = torch.cat([z_clusters, cf_tensor], dim=-1)  
            logits = self.actor_layer(z_with_cf).squeeze(-1)         

        # Step 5: Pad to full num_clusters
        if actual_clusters < self.num_clusters:
            pad_logits = torch.full((self.num_clusters - actual_clusters,), -1e9, device=device)
            logits = torch.cat([logits, pad_logits])

            pad_embed = torch.zeros(self.num_clusters - actual_clusters, self.hidden_dim, device=device)
            z_clusters = torch.cat([z_clusters, pad_embed], dim=0)

        # Flatten cluster embeddings
        z_flat = z_clusters.view(1, -1)   

        # ✅ NEW: Calculate Value for this specific agent's state
        value = self.critic_layer(z_flat)

        logits = logits.unsqueeze(0)       
        probs = F.softmax(logits, dim=-1)

        # ✅ CHANGED: Now returning 'value' alongside the rest
        return probs, value, z_flat, assignments, centroids

    # ✅ NEW: Helper method for the learner to efficiently evaluate cached embeddings
    def evaluate_value(self, z_flat):
        return self.critic_layer(z_flat)

    def actor(self, data, cached_assignments=None):
        # ✅ CHANGED: Ignore the value output here
        probs, _, _, assignments, centroids = self.forward(data, cached_assignments)
        return probs, assignments, centroids