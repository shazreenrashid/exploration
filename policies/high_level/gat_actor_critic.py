import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.cluster import KMeans
import numpy as np


class CentralCritic(nn.Module):
    def __init__(self, hidden_dim=64, num_agents=4, num_clusters=4):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * num_clusters * num_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_embeddings):
        # agent_embeddings: list of [1, num_clusters * hidden_dim] tensors, one per agent
        global_state = torch.cat(agent_embeddings, dim=-1)  # [1, num_clusters * hidden_dim * num_agents]
        return self.critic(global_state)


class GATActorCritic(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, num_clusters=4):
        super().__init__()
        self.num_clusters = num_clusters
        self.hidden_dim = hidden_dim

        # 1. Shared GAT Backbone
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True, edge_dim=1)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, edge_dim=1)

        # 2. Actor Head
        # Input is hidden_dim + 1 because we concatenate the explicit claimed_fraction
        # scalar to each cluster embedding before scoring. This gives the actor a direct,
        # numerically precise deconfliction signal with a one-hop path to the loss,
        # rather than relying on the GAT to encode it implicitly in 64 dims.
        self.actor_layer = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, cached_assignments=None):
        device = data.x.device
        x, edge_index = data.x, data.edge_index

        # Safely extract and reshape edge attributes for GATConv
        edge_attr = data.edge_attr
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # Step 1: GAT Encoder
        h = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
        h = self.conv2(h, edge_index, edge_attr=edge_attr)  # [N_nodes, hidden_dim]

        # Step 2: Clustering
        node_positions = data.pos.detach().cpu().numpy()

        if cached_assignments is None:
            # Replan step: Run full KMeans
            actual_clusters = min(len(node_positions), self.num_clusters)
            kmeans = KMeans(n_clusters=actual_clusters, n_init=5, random_state=None)
            assignments = kmeans.fit_predict(node_positions)
            centroids = kmeans.cluster_centers_
        else:
            # Cached step: Reuse frozen assignments, pad new nodes with -1
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
        # Also compute the claimed fraction per cluster directly from raw feature dim 6.
        # We use the raw input x rather than the GAT output h because we want the
        # explicit broadcast mask value, not a transformed version of it.
        assignments_tensor = torch.tensor(assignments, dtype=torch.long, device=device)
        cluster_embeddings = []
        claimed_fractions = []  # one scalar per cluster

        for c in range(actual_clusters):
            mask = (assignments_tensor == c)  # -1 entries are naturally excluded
            if mask.any():
                cluster_embeddings.append(h[mask].mean(dim=0))
                # Feature dim 6 is the broadcast mask: 1.0 if claimed by a peer, else 0.0
                claimed_fractions.append(x[mask, 6].mean().unsqueeze(0))
            else:
                cluster_embeddings.append(torch.zeros(self.hidden_dim, device=device))
                claimed_fractions.append(torch.zeros(1, device=device))

        # Step 4: Handle empty graph edge case
        if actual_clusters == 0:
            z_clusters = torch.zeros((0, self.hidden_dim), device=device)
            logits = torch.zeros((0,), device=device)
        else:
            z_clusters = torch.stack(cluster_embeddings)          # [actual_clusters, hidden_dim]
            cf_tensor = torch.stack(claimed_fractions)            # [actual_clusters, 1]

            # Concatenate claimed fraction onto each cluster embedding before scoring.
            # The actor head now has a direct, one-hop gradient path from the
            # deconfliction penalty to the weight that scales claimed_fraction.
            z_with_cf = torch.cat([z_clusters, cf_tensor], dim=-1)  # [actual_clusters, hidden_dim + 1]

            logits = self.actor_layer(z_with_cf).squeeze(-1)         # [actual_clusters]

        # Step 5: Pad to full num_clusters size so network shapes stay fixed
        if actual_clusters < self.num_clusters:
            pad_logits = torch.full((self.num_clusters - actual_clusters,), -1e9, device=device)
            logits = torch.cat([logits, pad_logits])

            pad_embed = torch.zeros(self.num_clusters - actual_clusters, self.hidden_dim, device=device)
            z_clusters = torch.cat([z_clusters, pad_embed], dim=0)

        # Flatten cluster embeddings for the Central Critic.
        # Note: z_flat uses z_clusters (without the claimed fraction) so the critic's
        # input dimension stays fixed at num_clusters * hidden_dim = 256 per agent.
        z_flat = z_clusters.view(1, -1)   # [1, num_clusters * hidden_dim]

        logits = logits.unsqueeze(0)       # [1, num_clusters]
        probs = F.softmax(logits, dim=-1)

        return probs, z_flat, assignments, centroids

    def actor(self, data, cached_assignments=None):
        probs, _, assignments, centroids = self.forward(data, cached_assignments)
        return probs, assignments, centroids
