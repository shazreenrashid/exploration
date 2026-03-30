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

# ── colour palette & global settings ──────────────────────────────────────────
AGENT_COLOURS   = ['#2196F3', '#4CAF50', '#9C27B0', '#FF9800']
CLUSTER_CMAPS   = ['tab10', 'tab10', 'tab10', 'tab10']
FRONTIER_MARK   = '*'
BREADCRUMB_MARK = 'o'

# Easily change the maximum number of steps for the evaluation here:
MAX_EVAL_STEPS  = 1500

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
# 2.  Run One Evaluation Episode
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

    # Uses the overridden max_steps from the config (which uses MAX_EVAL_STEPS)
    for step in range(config.get('max_steps', MAX_EVAL_STEPS)):
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
# 3.  Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def draw_episode(snapshots, save_path='episode_viz.png'):
    snap = snapshots[-1]
    num_agents = len(snap['agent_positions'])
    
    # ── Set up a simpler 1x2 figure ──────────────────────────────────────────
    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)

    def style_ax(ax, title, fontsize=12):
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='white', fontsize=fontsize, pad=10, fontweight='bold')
        ax.tick_params(colors='#aaaaaa', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    # ── Plot 1: True Environment ─────────────────────────────────────────────
    ax_world = fig.add_subplot(gs[0, 0])
    im = ax_world.imshow(snap['R_true'].T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=1)
    plt.colorbar(im, ax=ax_world, fraction=0.04, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')
    
    for i in range(num_agents):
        ax_world.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i], s=100, zorder=5, edgecolors='white', linewidths=1.2)
    style_ax(ax_world, 'Ground Truth Risk World')

    # ── Plot 2: All Agents (Trajectories) ────────────────────────────────────
    ax_traj = fig.add_subplot(gs[0, 1])
    # Show a faded version of the risk map as the background
    ax_traj.imshow(snap['R_true'].T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], alpha=0.35, vmin=0, vmax=1)
    
    for i in range(num_agents):
        traj = snap['trajectories'][i]
        if len(traj) > 1:
            ax_traj.plot(traj[:, 0], traj[:, 1], color=AGENT_COLOURS[i], linewidth=1.5, alpha=0.9, label=f'Agent {i}')
        # Draw the agent's current position
        ax_traj.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i], s=100, zorder=5, edgecolors='white', linewidths=1.2)
    
    ax_traj.legend(loc='upper right', fontsize=9, framealpha=0.4, labelcolor='white', facecolor='#22224a')
    # The title will now accurately reflect the final step count
    style_ax(ax_traj, f'Global Trajectories (Step {snap["step"] + 1})')

    plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n  Episode figure saved → {save_path}")
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 5.  Main 
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Visualise a trained checkpoint.')
    parser.add_argument('--checkpoint', type=int, default=None)
    parser.add_argument('--curve-only', action='store_true')
    parser.add_argument('--checkpoint-dir', default='checkpoints') 
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    config['train_mode'] = False
    
    # Force the maximum steps using the global variable
    config['max_steps'] = MAX_EVAL_STEPS 

    from environment.graph_environment import GraphEnvironment
    from agents.agent import Agent
    from policies.high_level.gat_actor_critic import CentralCritic

    env = GraphEnvironment(config, num_agents=config['num_agents'], grid_width=config['grid_width'], grid_height=config['grid_height'], risk_threshold=config['risk_threshold'])
    
    # If GraphEnvironment doesn't take max_steps in its constructor, we assign it directly to the environment instance
    if hasattr(env, 'max_steps'):
        env.max_steps = MAX_EVAL_STEPS

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
# AGENT_COLOURS  = ['#2196F3', '#4CAF50', '#9C27B0', '#FF9800']
# CLUSTER_CMAPS  = ['tab10', 'tab10', 'tab10', 'tab10']
# FRONTIER_MARK  = '*'
# BREADCRUMB_MARK = 'o'

# # ══════════════════════════════════════════════════════════════════════════════
# # 1.  Checkpoint Loading (CORRECTED LOGIC)
# # ══════════════════════════════════════════════════════════════════════════════

# def find_latest_episode(checkpoint_dir='checkpoints'):
#     # Search for anything starting with agent_0_ep (handles files or folders)
#     pattern = os.path.join(checkpoint_dir, 'agent_0_ep*')
#     files = sorted(glob.glob(pattern))
    
#     if not files:
#         raise FileNotFoundError(f"No checkpoints found in '{checkpoint_dir}/'")
    
#     # Get the last one, strip trailing slashes if it's a folder
#     latest_path = files[-1].rstrip(os.sep).rstrip('/')
#     # Extract number between 'ep' and the first dot (or end of string)
#     filename = os.path.basename(latest_path)
#     ep_str = filename.split('ep')[1].split('.')[0]
#     return int(ep_str)

# def load_checkpoint(agents, central_critic, ep_num, checkpoint_dir='checkpoints'):
#     def smart_load(path):
#         """Helper to load whether the path is a file or a folder."""
#         if os.path.isdir(path):
#             # If it's a directory, PyTorch often looks for 'data.pkl' or 'pytorch_model.bin' 
#             # or the directory itself if it's a saved zip state.
#             return torch.load(path, map_location=agents[0].device)
#         else:
#             return torch.load(path, map_location=agents[0].device)

#     for i, agent in enumerate(agents):
#         path = os.path.join(checkpoint_dir, f'agent_{i}_ep{ep_num}.pth')
#         if os.path.exists(path):
#             try:
#                 state_dict = smart_load(path)
#                 agent.policy.load_state_dict(state_dict)
#                 print(f"  Loaded agent {i} weights from episode {ep_num}")
#             except Exception as e:
#                 print(f"  ERROR loading agent {i} from {path}: {e}")
#         else:
#             print(f"  WARNING: {path} not found — agent {i} uses random weights")

#     critic_path = os.path.join(checkpoint_dir, f'central_critic_ep{ep_num}.pth')
#     if os.path.exists(critic_path):
#         try:
#             state_dict = smart_load(critic_path)
#             central_critic.load_state_dict(state_dict)
#             print(f"  Loaded central critic from episode {ep_num}")
#         except Exception as e:
#             print(f"  ERROR loading central critic: {e}")

# # ══════════════════════════════════════════════════════════════════════════════
# # 2.  Run One Evaluation Episode  (Unchanged)
# # ══════════════════════════════════════════════════════════════════════════════

# def run_episode(env, agents, config):
#     obs_buffers = env.reset()
#     for agent in agents:
#         agent.reset()

#     for i, agent in enumerate(agents):
#         agent.update_perception(obs_buffers[i], step=-1)
#     warmup_claimed = set()
#     for agent in agents:
#         agent.act(step=-1, train=False, claimed_cluster_ids=warmup_claimed)
#         if agent.current_cluster_id is not None:
#             warmup_claimed.add(agent.current_cluster_id)

#     snapshots = []
#     total_reward = 0.0

#     for step in range(5000):
#         for i, agent in enumerate(agents):
#             agent.update_perception(obs_buffers[i], step)

#         summaries = [a.get_broadcast_summary() for a in agents]
#         for agent in agents:
#             agent.belief.reset_broadcast_masks()
#             peer_summaries = [
#                 s for s in summaries
#                 if s is None or s['agent_id'] != agent.agent_id
#             ]
#             agent.belief.update_broadcast_masks(peer_summaries)

#         actions = {}
#         claimed = set()
#         with torch.no_grad():
#             for i, agent in enumerate(agents):
#                 target_pos = agent.act(step, train=False, claimed_cluster_ids=claimed)
#                 if agent.current_cluster_id is not None:
#                     claimed.add(agent.current_cluster_id)
#                 actions[i] = target_pos

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
# # 3.  Visualisation (Unchanged)
# # ══════════════════════════════════════════════════════════════════════════════

# def draw_episode(snapshots, save_path='episode_viz.png'):
#     snap = snapshots[-1]
#     num_agents = len(snap['agent_positions'])
    
#     # ── Set up a simpler 1x2 figure ──────────────────────────────────────────
#     fig = plt.figure(figsize=(14, 6))
#     fig.patch.set_facecolor('#1a1a2e')
#     gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.15)

#     def style_ax(ax, title, fontsize=12):
#         ax.set_facecolor('#16213e')
#         ax.set_title(title, color='white', fontsize=fontsize, pad=10, fontweight='bold')
#         ax.tick_params(colors='#aaaaaa', labelsize=9)
#         for spine in ax.spines.values():
#             spine.set_edgecolor('#444466')

#     # ── Plot 1: True Environment ─────────────────────────────────────────────
#     ax_world = fig.add_subplot(gs[0, 0])
#     im = ax_world.imshow(snap['R_true'].T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=1)
#     plt.colorbar(im, ax=ax_world, fraction=0.04, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')
    
#     for i in range(num_agents):
#         ax_world.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i], s=100, zorder=5, edgecolors='white', linewidths=1.2)
#     style_ax(ax_world, 'Ground Truth Risk World')

#     # ── Plot 2: All Agents (Trajectories) ────────────────────────────────────
#     ax_traj = fig.add_subplot(gs[0, 1])
#     # Show a faded version of the risk map as the background
#     ax_traj.imshow(snap['R_true'].T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], alpha=0.35, vmin=0, vmax=1)
    
#     for i in range(num_agents):
#         traj = snap['trajectories'][i]
#         if len(traj) > 1:
#             ax_traj.plot(traj[:, 0], traj[:, 1], color=AGENT_COLOURS[i], linewidth=1.5, alpha=0.9, label=f'Agent {i}')
#         # Draw the agent's current position
#         ax_traj.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i], s=100, zorder=5, edgecolors='white', linewidths=1.2)
    
#     ax_traj.legend(loc='upper right', fontsize=9, framealpha=0.4, labelcolor='white', facecolor='#22224a')
#     style_ax(ax_traj, f'Global Trajectories (Step {snap["step"]})')

#     plt.savefig(save_path, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
#     print(f"\n  Episode figure saved → {save_path}")
#     plt.show()

# # ══════════════════════════════════════════════════════════════════════════════
# # 5.  Main (Updated to use 'checkpoints' by default)
# # ══════════════════════════════════════════════════════════════════════════════

# def main():
#     parser = argparse.ArgumentParser(description='Visualise a trained checkpoint.')
#     parser.add_argument('--checkpoint', type=int, default=None)
#     parser.add_argument('--curve-only', action='store_true')
#     parser.add_argument('--checkpoint-dir', default='checkpoints') # Matches your folder name
#     parser.add_argument('--config', default='config.yaml')
#     args = parser.parse_args()

#     with open(args.config, 'r') as f:
#         config = yaml.safe_load(f)
#     config['train_mode'] = False

#     from environment.graph_environment import GraphEnvironment
#     from agents.agent import Agent
#     from policies.high_level.gat_actor_critic import CentralCritic

#     env = GraphEnvironment(num_agents=config['num_agents'], grid_width=config['grid_width'], grid_height=config['grid_height'], risk_threshold=config['risk_threshold'])
#     agents = [Agent(i, config) for i in range(config['num_agents'])]
#     central_critic = CentralCritic(hidden_dim=config.get('hidden_dim', 64), num_agents=config['num_agents'], num_clusters=config.get('num_clusters', 4)).to(agents[0].device)

#     ep_num = args.checkpoint
#     if ep_num is None:
#         ep_num = find_latest_episode(args.checkpoint_dir)
    
#     print(f"\nLoading checkpoint from episode {ep_num} ...")
#     load_checkpoint(agents, central_critic, ep_num, args.checkpoint_dir)

#     for agent in agents:
#         agent.policy.eval()
#     central_critic.eval()

#     snapshots, total_reward = run_episode(env, agents, config)
#     draw_episode(snapshots, save_path=f'episode_viz_ep{ep_num}.png')

# if __name__ == '__main__':
#     main()
