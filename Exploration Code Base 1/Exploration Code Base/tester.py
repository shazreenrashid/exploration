import argparse
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch
import yaml

# ── colour palette ────────────────────────────────────────────────────────────
AGENT_COLOURS  = ['#2196F3', '#4CAF50', '#9C27B0', '#FF9800']
CLUSTER_CMAPS  = ['tab10', 'tab10', 'tab10', 'tab10']
FRONTIER_MARK  = '*'
BREADCRUMB_MARK = 'o'

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Checkpoint Loading (CORRECTED LOGIC)
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_episode(checkpoint_dir='checkpoints'):
    # Search for anything starting with agent_0_ep (handles files or folders)
    pattern = os.path.join(checkpoint_dir, 'agent_0_ep*')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No checkpoints found in '{checkpoint_dir}/'")
    
    # Get the last one, strip trailing slashes if it's a folder
    latest_path = files[-1].rstrip(os.sep).rstrip('/')
    # Extract number between 'ep' and the first dot (or end of string)
    filename = os.path.basename(latest_path)
    ep_str = filename.split('ep')[1].split('.')[0]
    return int(ep_str)

def load_checkpoint(agents, central_critic, ep_num, checkpoint_dir='checkpoints'):
    def smart_load(path):
        """Helper to load whether the path is a file or a folder."""
        if os.path.isdir(path):
            # If it's a directory, PyTorch often looks for 'data.pkl' or 'pytorch_model.bin' 
            # or the directory itself if it's a saved zip state.
            return torch.load(path, map_location=agents[0].device)
        else:
            return torch.load(path, map_location=agents[0].device)

    for i, agent in enumerate(agents):
        path = os.path.join(checkpoint_dir, f'agent_{i}_ep{ep_num}.pth')
        if os.path.exists(path):
            try:
                state_dict = smart_load(path)
                agent.policy.load_state_dict(state_dict)
                print(f"  Loaded agent {i} weights from episode {ep_num}")
            except Exception as e:
                print(f"  ERROR loading agent {i} from {path}: {e}")
        else:
            print(f"  WARNING: {path} not found — agent {i} uses random weights")

    critic_path = os.path.join(checkpoint_dir, f'central_critic_ep{ep_num}.pth')
    if os.path.exists(critic_path):
        try:
            state_dict = smart_load(critic_path)
            central_critic.load_state_dict(state_dict)
            print(f"  Loaded central critic from episode {ep_num}")
        except Exception as e:
            print(f"  ERROR loading central critic: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  Run One Evaluation Episode  (Unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(env, agents, config):
    obs_buffers = env.reset()
    for agent in agents:
        agent.reset()

    for i, agent in enumerate(agents):
        agent.update_perception(obs_buffers[i], step=-1)
    warmup_claimed = set()
    for agent in agents:
        agent.act(step=-1, train=False, claimed_cluster_ids=warmup_claimed)
        if agent.current_cluster_id is not None:
            warmup_claimed.add(agent.current_cluster_id)

    snapshots = []
    total_reward = 0.0

    for step in range(config.get('max_steps', 1000)):
        for i, agent in enumerate(agents):
            agent.update_perception(obs_buffers[i], step)

        summaries = [a.get_broadcast_summary() for a in agents]
        for agent in agents:
            agent.belief.reset_broadcast_masks()
            peer_summaries = [
                s for s in summaries
                if s is None or s['agent_id'] != agent.agent_id
            ]
            agent.belief.update_broadcast_masks(peer_summaries)

        actions = {}
        claimed = set()
        with torch.no_grad():
            for i, agent in enumerate(agents):
                target_pos = agent.act(step, train=False, claimed_cluster_ids=claimed)
                if agent.current_cluster_id is not None:
                    claimed.add(agent.current_cluster_id)
                actions[i] = target_pos

        snap = {
            'step':              step,
            'agent_positions':   [a.current_pos.copy() for a in agents],
            'belief_R':          [a.belief.R.copy() for a in agents],
            'belief_C':          [a.belief.C.copy() for a in agents],
            'graphs':            [a.belief.graph.copy() for a in agents],
            'assignments':       [a.cached_assignments.copy()
                                  if a.cached_assignments is not None
                                  else np.array([]) for a in agents],
            'cluster_ids':       [a.current_cluster_id for a in agents],
            'centroids':         [a.target_cluster_centroid.copy()
                                  if a.target_cluster_centroid is not None
                                  else None for a in agents],
            'trajectories':      [np.array(env.agent_trajectories[i]).copy()
                                  for i in range(len(agents))],
            'R_true':            env.R_true.copy(),
            'global_coverage':   env.global_coverage.copy(),
        }
        snapshots.append(snap)

        obs_buffers, reward, done, _ = env.step(actions)
        total_reward += reward
        if done:
            break

    print(f"\n  Episode finished — {len(snapshots)} steps | Total reward: {total_reward:.2f}")
    return snapshots, total_reward

# ══════════════════════════════════════════════════════════════════════════════
# 3.  Visualisation (Unchanged)
# ══════════════════════════════════════════════════════════════════════════════

def draw_episode(snapshots, save_path='episode_viz.png'):
    snap = snapshots[-1]
    num_agents = len(snap['agent_positions'])
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.38, wspace=0.28, top=0.93, bottom=0.04, left=0.04, right=0.97)

    def style_ax(ax, title, fontsize=11):
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='white', fontsize=fontsize, pad=6, fontweight='bold')
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    ax_world = fig.add_subplot(gs[0, 0])
    im = ax_world.imshow(snap['R_true'].T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_world, fraction=0.04, pad=0.02).ax.yaxis.set_tick_params(color='white', labelcolor='white')
    for i in range(num_agents):
        ax_world.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i], s=80, zorder=5, edgecolors='white', linewidths=0.8)
    style_ax(ax_world, 'Ground Truth Risk World')

    ax_traj = fig.add_subplot(gs[0, 1])
    ax_traj.imshow(snap['R_true'].T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], alpha=0.35, vmin=0, vmax=1)
    for i in range(num_agents):
        traj = snap['trajectories'][i]
        if len(traj) > 1:
            ax_traj.plot(traj[:, 0], traj[:, 1], color=AGENT_COLOURS[i], linewidth=1.2, alpha=0.9, label=f'Agent {i}')
        ax_traj.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i], s=80, zorder=5, edgecolors='white', linewidths=0.8)
    style_ax(ax_traj, f'Global Trajectories  (step {snap["step"]})')

    ax_cov = fig.add_subplot(gs[0, 2])
    cov_display = snap['global_coverage'].astype(float)
    ax_cov.imshow(cov_display.T, cmap='Blues', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=1)
    covered_pct = 100.0 * snap['global_coverage'].sum() / (100 * 100)
    style_ax(ax_cov, f'Team Coverage  ({covered_pct:.1f}% explored)')

    ax_leg = fig.add_subplot(gs[0, 3])
    ax_leg.axis('off')
    legend_items = []
    for i in range(num_agents):
        legend_items.append(mpatches.Patch(color=AGENT_COLOURS[i], label=f'Agent {i}'))
    legend_items += [
        plt.Line2D([0], [0], marker=FRONTIER_MARK, color='w', linestyle='None', markersize=10, label='Frontier', markerfacecolor='yellow'),
        plt.Line2D([0], [0], marker=BREADCRUMB_MARK, color='w', linestyle='None', markersize=7, label='Breadcrumb', markerfacecolor='white'),
    ]
    ax_leg.legend(handles=legend_items, loc='center', fontsize=9, framealpha=0.0, labelcolor='white')

    from environment.node_types import NodeType
    cluster_colours = plt.get_cmap('tab10')
    for i in range(num_agents):
        graph = snap['graphs'][i]
        R = snap['belief_R'][i]
        assignments = snap['assignments'][i]
        pos_agent = snap['agent_positions'][i]
        node_ids = list(graph.nodes)

        ax_b = fig.add_subplot(gs[1, i])
        ax_b.imshow(R.T, cmap='YlOrRd', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=1)
        for u, v in graph.edges():
            p1, p2 = graph.nodes[u]['pos'], graph.nodes[v]['pos']
            ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]], color='white', alpha=0.25, linewidth=0.6)
        style_ax(ax_b, f'Agent {i} Belief')

        ax_c = fig.add_subplot(gs[2, i])
        style_ax(ax_c, f'Agent {i} Clusters')

    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main (Updated to use 'checkpoints' by default)
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Visualise a trained checkpoint.')
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--curve-only', action='store_true')
    parser.add_argument('--checkpoint-dir', default='checkpoints') # Matches your folder name
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['train_mode'] = False

    from environment.graph_environment import GraphEnvironment
    from agents.agent import Agent
    from policies.high_level.gat_actor_critic import CentralCritic

    env = GraphEnvironment(num_agents=config['num_agents'], grid_width=config['grid_width'], grid_height=config['grid_height'], risk_threshold=config['risk_threshold'])
    agents = [Agent(i, config) for i in range(config['num_agents'])]
    central_critic = CentralCritic(hidden_dim=config.get('hidden_dim', 64), num_agents=config['num_agents'], num_clusters=config.get('num_clusters', 4)).to(agents[0].device)

    ep_num = args.checkpoint
    if ep_num is None:
        ep_num = find_latest_episode(args.checkpoint_dir)
    
    print(f"\nLoading checkpoint from episode {ep_num} ...")
    load_checkpoint(agents, central_critic, ep_num, args.checkpoint_dir)

    for agent in agents:
        agent.policy.eval()
    central_critic.eval()

    snapshots, total_reward = run_episode(env, agents, config)
    draw_episode(snapshots, save_path=f'episode_viz_ep{ep_num}.png')

if __name__ == '__main__':
    main()



# """
# tester.py — Load a checkpoint and visualise a full episode.

# Usage:
#     python tester.py                        # loads latest checkpoint automatically
#     python tester.py --checkpoint 200       # loads episode 200 checkpoint
#     python tester.py --curve-only           # just plot the training curve, no episode run

# Produces two figures:
#     Figure 1: Episode visualisation (true world, trajectories, per-agent belief maps,
#               per-agent graphs with frontier/breadcrumb nodes, per-agent cluster maps)
#     Figure 2: Training curve (reward vs episode)
# """

# import argparse
# import glob
# import os
# import json

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.gridspec as gridspec
# import torch
# import yaml

# # ── colour palette ────────────────────────────────────────────────────────────
# AGENT_COLOURS  = ['#2196F3', '#4CAF50', '#9C27B0', '#FF9800']   # blue, green, purple, orange
# CLUSTER_CMAPS  = ['tab10', 'tab10', 'tab10', 'tab10']            # same cmap for cluster IDs across agents
# FRONTIER_MARK  = '*'   # star
# BREADCRUMB_MARK = 'o'  # circle


# # ══════════════════════════════════════════════════════════════════════════════
# # 1.  Checkpoint Loading
# # ══════════════════════════════════════════════════════════════════════════════

# def find_latest_episode(checkpoint_dir='ckpts'):
#     files = sorted(glob.glob(os.path.join(checkpoint_dir, 'agent_0_ep*.pth')))
#     if not files:
#         raise FileNotFoundError(f"No checkpoints found in '{checkpoint_dir}/'")
#     return int(files[-1].split('ep')[1].split('.')[0])


# def load_checkpoint(agents, central_critic, ep_num, checkpoint_dir='ckpts'):
#     for i, agent in enumerate(agents):
#         path = os.path.join(checkpoint_dir, f'agent_{i}_ep{ep_num}.pth')
#         if os.path.exists(path):
#             agent.policy.load_state_dict(
#                 torch.load(path, map_location=agent.device)
#             )
#             print(f"  Loaded agent {i} weights from episode {ep_num}")
#         else:
#             print(f"  WARNING: {path} not found — agent {i} uses random weights")

#     critic_path = os.path.join(checkpoint_dir, f'central_critic_ep{ep_num}.pth')
#     if os.path.exists(critic_path):
#         central_critic.load_state_dict(
#             torch.load(critic_path, map_location=agents[0].device)
#         )
#         print(f"  Loaded central critic from episode {ep_num}")


# # ══════════════════════════════════════════════════════════════════════════════
# # 2.  Run One Evaluation Episode  (no training, no gradient)
# # ══════════════════════════════════════════════════════════════════════════════

# def run_episode(env, agents, config):
#     """
#     Runs one complete evaluation episode.
#     Returns a list of per-step snapshots — each snapshot is a dict containing
#     everything needed to recreate the visualisation at that step.
#     """
#     obs_buffers = env.reset()
#     for agent in agents:
#         agent.reset()

#     # Warm-up
#     for i, agent in enumerate(agents):
#         agent.update_perception(obs_buffers[i], step=-1)
#     warmup_claimed = set()
#     for agent in agents:
#         agent.act(step=-1, train=False, claimed_cluster_ids=warmup_claimed)
#         if agent.current_cluster_id is not None:
#             warmup_claimed.add(agent.current_cluster_id)

#     snapshots = []
#     total_reward = 0.0

#     for step in range(config.get('max_steps', 200)):
#         # Perception
#         for i, agent in enumerate(agents):
#             agent.update_perception(obs_buffers[i], step)

#         # Coordination
#         summaries = [a.get_broadcast_summary() for a in agents]
#         for agent in agents:
#             agent.belief.reset_broadcast_masks()
#             peer_summaries = [
#                 s for s in summaries
#                 if s is None or s['agent_id'] != agent.agent_id
#             ]
#             agent.belief.update_broadcast_masks(peer_summaries)

#         # Act (sequential, greedy, no grad)
#         actions = {}
#         claimed = set()
#         with torch.no_grad():
#             for i, agent in enumerate(agents):
#                 target_pos = agent.act(step, train=False, claimed_cluster_ids=claimed)
#                 if agent.current_cluster_id is not None:
#                     claimed.add(agent.current_cluster_id)
#                 actions[i] = target_pos

#         # Save snapshot AFTER acting so graphs are populated
#         snap = {
#             'step':              step,
#             'agent_positions':   [a.current_pos.copy() for a in agents],
#             'belief_R':          [a.belief.R.copy() for a in agents],
#             'belief_C':          [a.belief.C.copy() for a in agents],
#             'graphs':            [a.belief.graph.copy() for a in agents],
#             'assignments':       [a.cached_assignments.copy()
#                                   if a.cached_assignments is not None
#                                   else np.array([]) for a in agents],
#             'cluster_ids':       [a.current_cluster_id for a in agents],
#             'centroids':         [a.target_cluster_centroid.copy()
#                                   if a.target_cluster_centroid is not None
#                                   else None for a in agents],
#             'trajectories':      [np.array(env.agent_trajectories[i]).copy()
#                                   for i in range(len(agents))],
#             'R_true':            env.R_true.copy(),
#             'global_coverage':   env.global_coverage.copy(),
#         }
#         snapshots.append(snap)

#         obs_buffers, reward, done, _ = env.step(actions)
#         total_reward += reward

#         if done:
#             break

#     print(f"\n  Episode finished — {len(snapshots)} steps | Total reward: {total_reward:.2f}")
#     return snapshots, total_reward


# # ══════════════════════════════════════════════════════════════════════════════
# # 3.  Visualisation  — Episode Figure
# # ══════════════════════════════════════════════════════════════════════════════

# def draw_episode(snapshots, save_path='episode_viz.png'):
#     """
#     Draws a rich 3-row figure using the FINAL snapshot of the episode.

#     Layout (3 rows × 4 cols, some cells merged):
#     ┌──────────────┬──────────────┬──────────────┬──────────────┐
#     │  True World  │  Trajectories│  Coverage    │  (legend)    │  Row 0
#     ├──────────────┼──────────────┼──────────────┼──────────────┤
#     │  Agent 0     │  Agent 1     │  Agent 2     │  Agent 3     │  Row 1
#     │  Belief +    │  Belief +    │  Belief +    │  Belief +    │
#     │  Graph       │  Graph       │  Graph       │  Graph       │
#     ├──────────────┼──────────────┼──────────────┼──────────────┤
#     │  Agent 0     │  Agent 1     │  Agent 2     │  Agent 3     │  Row 2
#     │  Clusters    │  Clusters    │  Clusters    │  Clusters    │
#     └──────────────┴──────────────┴──────────────┴──────────────┘
#     """
#     snap = snapshots[-1]          # use final state of the episode
#     num_agents = len(snap['agent_positions'])

#     fig = plt.figure(figsize=(22, 16))
#     fig.patch.set_facecolor('#1a1a2e')

#     gs = gridspec.GridSpec(
#         3, 4,
#         figure=fig,
#         hspace=0.38,
#         wspace=0.28,
#         top=0.93, bottom=0.04, left=0.04, right=0.97
#     )

#     # ── shared style helper ───────────────────────────────────────────────
#     def style_ax(ax, title, fontsize=11):
#         ax.set_facecolor('#16213e')
#         ax.set_title(title, color='white', fontsize=fontsize, pad=6, fontweight='bold')
#         ax.tick_params(colors='#aaaaaa', labelsize=7)
#         for spine in ax.spines.values():
#             spine.set_edgecolor('#444466')

#     # ── Row 0, col 0: True risk world ────────────────────────────────────
#     ax_world = fig.add_subplot(gs[0, 0])
#     im = ax_world.imshow(
#         snap['R_true'].T, cmap='hot', origin='lower',
#         extent=[0, 100, 0, 100], vmin=0, vmax=1
#     )
#     plt.colorbar(im, ax=ax_world, fraction=0.04, pad=0.02).ax.yaxis.set_tick_params(color='white', labelcolor='white')
#     for i in range(num_agents):
#         ax_world.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i],
#                          s=80, zorder=5, edgecolors='white', linewidths=0.8)
#     style_ax(ax_world, 'Ground Truth Risk World')

#     # ── Row 0, col 1: Global trajectories ────────────────────────────────
#     ax_traj = fig.add_subplot(gs[0, 1])
#     ax_traj.imshow(
#         snap['R_true'].T, cmap='hot', origin='lower',
#         extent=[0, 100, 0, 100], alpha=0.35, vmin=0, vmax=1
#     )
#     for i in range(num_agents):
#         traj = snap['trajectories'][i]
#         if len(traj) > 1:
#             ax_traj.plot(traj[:, 0], traj[:, 1], color=AGENT_COLOURS[i],
#                          linewidth=1.2, alpha=0.9, label=f'Agent {i}')
#         ax_traj.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i],
#                         s=80, zorder=5, edgecolors='white', linewidths=0.8)
#     ax_traj.legend(loc='upper right', fontsize=7, framealpha=0.4,
#                    labelcolor='white', facecolor='#22224a')
#     style_ax(ax_traj, f'Global Trajectories  (step {snap["step"]})')

#     # ── Row 0, col 2: Team coverage map ──────────────────────────────────
#     ax_cov = fig.add_subplot(gs[0, 2])
#     cov_display = snap['global_coverage'].astype(float)
#     ax_cov.imshow(
#         cov_display.T, cmap='Blues', origin='lower',
#         extent=[0, 100, 0, 100], vmin=0, vmax=1
#     )
#     covered_pct = 100.0 * snap['global_coverage'].sum() / (100 * 100)
#     style_ax(ax_cov, f'Team Coverage  ({covered_pct:.1f}% explored)')

#     # ── Row 0, col 3: Legend panel ────────────────────────────────────────
#     ax_leg = fig.add_subplot(gs[0, 3])
#     ax_leg.set_facecolor('#16213e')
#     ax_leg.axis('off')
#     ax_leg.set_title('Legend', color='white', fontsize=11, pad=6, fontweight='bold')

#     legend_items = []
#     for i in range(num_agents):
#         legend_items.append(mpatches.Patch(color=AGENT_COLOURS[i], label=f'Agent {i}'))
#     legend_items += [
#         plt.Line2D([0], [0], marker=FRONTIER_MARK,  color='w', linestyle='None',
#                    markersize=10, label='Frontier node',   markerfacecolor='yellow'),
#         plt.Line2D([0], [0], marker=BREADCRUMB_MARK, color='w', linestyle='None',
#                    markersize=7,  label='Breadcrumb node', markerfacecolor='white',
#                    markeredgecolor='grey'),
#         plt.Line2D([0], [0], color='white', linewidth=1.5, alpha=0.4, label='Graph edge'),
#         mpatches.Patch(color='#880000', label='High risk (belief)'),
#         mpatches.Patch(color='#ffeeaa', label='Low risk (belief)'),
#         mpatches.Patch(color='#334455', label='Unknown (belief)'),
#         mpatches.Patch(facecolor='none', edgecolor='cyan',
#                        linewidth=1.5, linestyle='--', label='Cluster centroid'),
#     ]
#     ax_leg.legend(handles=legend_items, loc='center', fontsize=9,
#                   framealpha=0.0, labelcolor='white', handlelength=1.8,
#                   borderpad=1.2, labelspacing=0.9)

#     # ── Rows 1 & 2: Per-agent subplots ───────────────────────────────────
#     cluster_colours = plt.cm.get_cmap('tab10')

#     for i in range(num_agents):
#         graph       = snap['graphs'][i]
#         R           = snap['belief_R'][i]
#         assignments = snap['assignments'][i]
#         pos_agent   = snap['agent_positions'][i]
#         centroid    = snap['centroids'][i]
#         cur_cluster = snap['cluster_ids'][i]
#         node_ids    = list(graph.nodes)

#         # ── Row 1: Belief map + graph overlay ────────────────────────────
#         ax_b = fig.add_subplot(gs[1, i])

#         # Risk belief matrix — use a diverging-like map centred at 0.5 (unknown)
#         im_b = ax_b.imshow(
#             R.T, cmap='YlOrRd', origin='lower',
#             extent=[0, 100, 0, 100], vmin=0, vmax=1
#         )
#         plt.colorbar(im_b, ax=ax_b, fraction=0.04, pad=0.02).ax.yaxis.set_tick_params(
#             color='white', labelcolor='white')

#         # Graph edges
#         for u, v in graph.edges():
#             p1 = graph.nodes[u]['pos']
#             p2 = graph.nodes[v]['pos']
#             ax_b.plot([p1[0], p2[0]], [p1[1], p2[1]],
#                       color='white', alpha=0.25, linewidth=0.6, zorder=2)

#         # Frontier nodes (stars) and Breadcrumb nodes (circles)
#         from environment.node_types import NodeType
#         for n_id in node_ids:
#             npos  = graph.nodes[n_id]['pos']
#             ntype = graph.nodes[n_id]['type']
#             if ntype == NodeType.FRONTIER:
#                 ax_b.scatter(npos[0], npos[1], marker=FRONTIER_MARK,
#                              color='yellow', s=70, zorder=4, linewidths=0)
#             else:
#                 ax_b.scatter(npos[0], npos[1], marker=BREADCRUMB_MARK,
#                              color='white', s=18, alpha=0.6, zorder=3, linewidths=0)

#         # Agent current position
#         ax_b.scatter(*pos_agent, color=AGENT_COLOURS[i], s=120,
#                      zorder=6, edgecolors='white', linewidths=1.2, marker='D')

#         style_ax(ax_b, f'Agent {i} — Belief Map + Graph')

#         # ── Row 2: Cluster assignment map ────────────────────────────────
#         ax_c = fig.add_subplot(gs[2, i])
#         ax_c.set_facecolor('#0d0d1a')

#         # Background: coverage mask so unknown area is clearly dark
#         C_mask = snap['belief_C'][i]
#         ax_c.imshow(
#             C_mask.T, cmap='binary', origin='lower',
#             extent=[0, 100, 0, 100], vmin=0, vmax=1, alpha=0.15
#         )

#         # Draw graph edges in agent colour
#         for u, v in graph.edges():
#             p1 = graph.nodes[u]['pos']
#             p2 = graph.nodes[v]['pos']
#             ax_c.plot([p1[0], p2[0]], [p1[1], p2[1]],
#                       color=AGENT_COLOURS[i], alpha=0.2, linewidth=0.6, zorder=2)

#         # Draw nodes coloured by cluster ID
#         if len(assignments) == len(node_ids) and len(node_ids) > 0:
#             for k, n_id in enumerate(node_ids):
#                 npos   = graph.nodes[n_id]['pos']
#                 ntype  = graph.nodes[n_id]['type']
#                 c_id   = int(assignments[k])
#                 colour = cluster_colours(c_id / 9.0)   # tab10 has 10 colours
#                 alpha  = 1.0 if c_id == cur_cluster else 0.45

#                 if ntype == NodeType.FRONTIER:
#                     ax_c.scatter(npos[0], npos[1], marker=FRONTIER_MARK,
#                                  color=colour, s=90, zorder=4, alpha=alpha,
#                                  edgecolors='white' if c_id == cur_cluster else 'none',
#                                  linewidths=0.6)
#                 else:
#                     ax_c.scatter(npos[0], npos[1], marker=BREADCRUMB_MARK,
#                                  color=colour, s=22, zorder=3, alpha=alpha * 0.8)
#         else:
#             # Assignments not yet populated — draw all nodes in agent colour
#             for n_id in node_ids:
#                 npos = graph.nodes[n_id]['pos']
#                 ax_c.scatter(npos[0], npos[1], color=AGENT_COLOURS[i],
#                              s=25, zorder=3, alpha=0.7)

#         # Draw current cluster centroid with a dashed circle
#         if centroid is not None:
#             circle = plt.Circle(centroid, radius=12, color='cyan',
#                                 fill=False, linestyle='--', linewidth=1.4,
#                                 alpha=0.7, zorder=5)
#             ax_c.add_patch(circle)
#             ax_c.scatter(*centroid, color='cyan', s=60, zorder=6,
#                          marker='+', linewidths=1.5)

#         # Agent position
#         ax_c.scatter(*pos_agent, color=AGENT_COLOURS[i], s=120,
#                      zorder=7, edgecolors='white', linewidths=1.2, marker='D')

#         ax_c.set_xlim(0, 100)
#         ax_c.set_ylim(0, 100)

#         # Cluster legend for this agent
#         if len(assignments) > 0:
#             unique_clusters = sorted(np.unique(assignments).tolist())
#             patches = [
#                 mpatches.Patch(
#                     color=cluster_colours(int(c) / 9.0),
#                     label=f'C{int(c)}' + (' ◀' if int(c) == cur_cluster else '')
#                 )
#                 for c in unique_clusters
#             ]
#             ax_c.legend(handles=patches, loc='upper right', fontsize=6.5,
#                         framealpha=0.5, labelcolor='white', facecolor='#22224a',
#                         handlelength=1, borderpad=0.6, labelspacing=0.5)

#         style_ax(ax_c, f'Agent {i} — Cluster Assignments')

#     fig.suptitle(
#         f'Multi-Agent Exploration — Final State  |  Step {snap["step"]}',
#         color='white', fontsize=14, fontweight='bold', y=0.97
#     )

#     plt.savefig(save_path, dpi=130, bbox_inches='tight',
#                 facecolor=fig.get_facecolor())
#     print(f"\n  Episode figure saved → {save_path}")
#     plt.show()


# # ══════════════════════════════════════════════════════════════════════════════
# # 4.  Training Curve
# # ══════════════════════════════════════════════════════════════════════════════

# def plot_training_curve(rewards_path='reward_log.json', save_path='training_curve_plot.png'):
#     """
#     Loads reward_log.json (written by main.py — see note below) and plots:
#       - Raw episode reward
#       - 20-episode rolling mean
#       - 20-episode rolling std band
    
#     NOTE: To enable this, add these two lines to main.py inside the training loop
#     just after reward_history.append(episode_reward):

#         with open('reward_log.json', 'w') as f:
#             json.dump(reward_history, f)
#     """
#     if not os.path.exists(rewards_path):
#         print(f"\n  WARNING: '{rewards_path}' not found.")
#         print("  Add this to main.py after reward_history.append(episode_reward):")
#         print("      import json")
#         print("      with open('reward_log.json', 'w') as f:")
#         print("          json.dump(reward_history, f)")
#         return

#     with open(rewards_path, 'r') as f:
#         rewards = np.array(json.load(f))

#     episodes = np.arange(len(rewards))
#     window   = min(20, len(rewards))

#     # Rolling stats
#     rolling_mean = np.convolve(rewards, np.ones(window) / window, mode='valid')
#     rolling_std  = np.array([
#         rewards[max(0, j - window):j].std()
#         for j in range(window, len(rewards) + 1)
#     ])
#     roll_ep = episodes[window - 1:]

#     fig, axes = plt.subplots(1, 2, figsize=(16, 5))
#     fig.patch.set_facecolor('#1a1a2e')

#     for ax in axes:
#         ax.set_facecolor('#16213e')
#         ax.tick_params(colors='#aaaaaa')
#         for spine in ax.spines.values():
#             spine.set_edgecolor('#444466')

#     # ── Left panel: raw + rolling mean ───────────────────────────────────
#     axes[0].plot(episodes, rewards, color='#4fc3f7', alpha=0.35,
#                  linewidth=0.8, label='Episode reward')
#     axes[0].plot(roll_ep, rolling_mean, color='#ff8a65', linewidth=2.2,
#                  label=f'{window}-ep rolling mean')
#     axes[0].fill_between(
#         roll_ep,
#         rolling_mean - rolling_std,
#         rolling_mean + rolling_std,
#         color='#ff8a65', alpha=0.15, label='±1 std'
#     )
#     axes[0].axhline(0, color='#666688', linewidth=0.8, linestyle='--')
#     axes[0].set_xlabel('Episode', color='white', fontsize=11)
#     axes[0].set_ylabel('Team Reward', color='white', fontsize=11)
#     axes[0].set_title('Training Curve', color='white', fontsize=13, fontweight='bold')
#     axes[0].legend(fontsize=9, framealpha=0.4, labelcolor='white',
#                    facecolor='#22224a')
#     axes[0].xaxis.label.set_color('white')
#     axes[0].yaxis.label.set_color('white')
#     axes[0].tick_params(axis='x', colors='#aaaaaa')
#     axes[0].tick_params(axis='y', colors='#aaaaaa')

#     # ── Right panel: cumulative max (best ever seen) ──────────────────────
#     cum_max = np.maximum.accumulate(rolling_mean)
#     axes[1].plot(roll_ep, rolling_mean, color='#4fc3f7', linewidth=1.5,
#                  alpha=0.6, label=f'{window}-ep rolling mean')
#     axes[1].plot(roll_ep, cum_max, color='#aed581', linewidth=2.2,
#                  linestyle='--', label='Best rolling mean so far')
#     axes[1].set_xlabel('Episode', color='white', fontsize=11)
#     axes[1].set_ylabel('Rolling Mean Reward', color='white', fontsize=11)
#     axes[1].set_title('Best Performance Achieved', color='white',
#                       fontsize=13, fontweight='bold')
#     axes[1].legend(fontsize=9, framealpha=0.4, labelcolor='white',
#                    facecolor='#22224a')
#     axes[1].xaxis.label.set_color('white')
#     axes[1].yaxis.label.set_color('white')
#     axes[1].tick_params(axis='x', colors='#aaaaaa')
#     axes[1].tick_params(axis='y', colors='#aaaaaa')

#     # Summary stats
#     print(f"\n  Training summary over {len(rewards)} episodes:")
#     print(f"    First 10 ep mean reward : {rewards[:10].mean():.2f}")
#     print(f"    Last  10 ep mean reward : {rewards[-10:].mean():.2f}")
#     print(f"    Best single episode     : {rewards.max():.2f}  (ep {rewards.argmax()})")
#     print(f"    Worst single episode    : {rewards.min():.2f}  (ep {rewards.argmin()})")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=130, bbox_inches='tight',
#                 facecolor=fig.get_facecolor())
#     print(f"  Training curve saved → {save_path}")
#     plt.show()


# # ══════════════════════════════════════════════════════════════════════════════
# # 5.  Main
# # ══════════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(description='Visualise a trained checkpoint.')
#     parser.add_argument('--checkpoint',  type=int,  default=None,
#                         help='Episode number to load (default: latest)')
#     parser.add_argument('--curve-only',  action='store_true',
#                         help='Only plot the training curve, skip episode run')
#     parser.add_argument('--checkpoint-dir', default='checkpoints')
#     parser.add_argument('--rewards-log',    default='reward_log.json')
#     parser.add_argument('--config',         default='config.yaml')
#     args = parser.parse_args()

#     # ── Training curve (always shown unless --curve-only skips episode) ──
#     # plot_training_curve(
#     #     rewards_path=args.rewards_log,
#     #     save_path='training_curve_plot.png'
#     # )

#     if args.curve_only:
#         return

#     # ── Load config ──────────────────────────────────────────────────────
#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
#     config['train_mode'] = False   # force eval mode

#     # ── Build environment and agents ─────────────────────────────────────
#     from environment.graph_environment import GraphEnvironment
#     from agents.agent import Agent
#     from policies.high_level.gat_actor_critic import CentralCritic

#     env = GraphEnvironment(
#         num_agents=config['num_agents'],
#         grid_width=config['grid_width'],
#         grid_height=config['grid_height'],
#         risk_threshold=config['risk_threshold']
#     )

#     agents = [Agent(i, config) for i in range(config['num_agents'])]
#     central_critic = CentralCritic(
#         hidden_dim=config.get('hidden_dim', 64),
#         num_agents=config['num_agents'],
#         num_clusters=config.get('num_clusters', 4)
#     ).to(agents[0].device)

#     # ── Load checkpoint ──────────────────────────────────────────────────
#     ep_num = args.checkpoint
#     if ep_num is None:
#         ep_num = find_latest_episode(args.checkpoint_dir)
#     print(f"\nLoading checkpoint from episode {ep_num} ...")
#     load_checkpoint(agents, central_critic, ep_num, args.checkpoint_dir)

#     # Set all policies to eval mode
#     for agent in agents:
#         agent.policy.eval()
#     central_critic.eval()

#     # ── Run evaluation episode ───────────────────────────────────────────
#     print("\nRunning evaluation episode ...")
#     snapshots, total_reward = run_episode(env, agents, config)

#     # ── Draw visualisation ───────────────────────────────────────────────
#     draw_episode(snapshots, save_path=f'episode_viz_ep{ep_num}.png')


# if __name__ == '__main__':
#     main()
