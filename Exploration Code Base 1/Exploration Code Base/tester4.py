import argparse
import glob
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
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
# 1.  Checkpoint Loading
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_episode(checkpoint_dir='checkpoints'):
    pattern = os.path.join(checkpoint_dir, 'agent_0_ep*')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No checkpoints found in '{checkpoint_dir}/'")
    
    latest_path = files[-1].rstrip(os.sep).rstrip('/')
    filename = os.path.basename(latest_path)
    ep_str = filename.split('ep')[1].split('.')[0]
    return int(ep_str)

def load_checkpoint(agents, central_critic, ep_num, checkpoint_dir='checkpoints'):
    def smart_load(path):
        if os.path.isdir(path):
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
# 3.  Simplified 3-Plot Animation
# ══════════════════════════════════════════════════════════════════════════════

def animate_episode(snapshots, save_path='episode_video.mp4'):
    print(f"\n  Generating 3-plot animation for {len(snapshots)} frames...")
    
    num_agents = len(snapshots[0]['agent_positions'])
    
    # Set up a clean 1x3 grid
    fig = plt.figure(figsize=(20, 6))
    fig.patch.set_facecolor('#1a1a2e')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.15)

    def style_ax(ax, title, fontsize=12):
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='white', fontsize=fontsize, pad=10, fontweight='bold')
        ax.tick_params(colors='#aaaaaa', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    # ── Initialize Axes ──────────────────────────────────────────────────────
    ax_world   = fig.add_subplot(gs[0, 0])
    ax_traj    = fig.add_subplot(gs[0, 1])
    ax_cluster = fig.add_subplot(gs[0, 2])

    # ── Setup Image Backgrounds and Colorbars ONCE ───────────────────────────
    snap0 = snapshots[0]
    
    # Plot 1: True World
    im_world = ax_world.imshow(snap0['R_true'].T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], vmin=0, vmax=1)
    plt.colorbar(im_world, ax=ax_world, fraction=0.04, pad=0.04).ax.yaxis.set_tick_params(color='white', labelcolor='white')
    
    # Plot 2: Trajectories
    im_traj = ax_traj.imshow(snap0['R_true'].T, cmap='hot', origin='lower', extent=[0, 100, 0, 100], alpha=0.35, vmin=0, vmax=1)
    
    # Plot 3: Combined Clusters (faded background)
    im_cluster = ax_cluster.imshow(snap0['R_true'].T, cmap='binary', origin='lower', extent=[0, 100, 0, 100], alpha=0.15, vmin=0, vmax=1)

    from environment.node_types import NodeType
    cluster_colours = plt.get_cmap('tab10')
    all_axes_to_clear = [ax_world, ax_traj, ax_cluster]

    # ── The Update Engine ────────────────────────────────────────────────────
    def update(frame):
        snap = snapshots[frame]
        
        # 1. Update backgrounds
        im_world.set_data(snap['R_true'].T)
        im_traj.set_data(snap['R_true'].T)
        im_cluster.set_data(snap['R_true'].T)

        # 2. Safely remove old vector elements to prevent memory leaks
        for ax in all_axes_to_clear:
            for coll in list(ax.collections): coll.remove()
            for line in list(ax.lines): line.remove()
            for patch in list(ax.patches): patch.remove()

        # 3. Redraw dynamic elements
        
        # ── Plot 1: World ──
        for i in range(num_agents):
            ax_world.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i], s=100, zorder=5, edgecolors='white', linewidths=1.0)
        style_ax(ax_world, 'Ground Truth Risk World')
        
        # ── Plot 2: Trajectories ──
        for i in range(num_agents):
            traj = snap['trajectories'][i]
            if len(traj) > 1:
                ax_traj.plot(traj[:, 0], traj[:, 1], color=AGENT_COLOURS[i], linewidth=1.5, alpha=0.9)
            ax_traj.scatter(*snap['agent_positions'][i], color=AGENT_COLOURS[i], s=100, zorder=5, edgecolors='white', linewidths=1.0)
        style_ax(ax_traj, f'Global Trajectories')

        # ── Plot 3: Combined Clusters ──
        for i in range(num_agents):
            graph       = snap['graphs'][i]
            assignments = snap['assignments'][i]
            pos_agent   = snap['agent_positions'][i]
            centroid    = snap['centroids'][i]
            cur_cluster = snap['cluster_ids'][i]
            node_ids    = list(graph.nodes)

            # Draw Graph Edges
            for u, v in graph.edges():
                p1, p2 = graph.nodes[u]['pos'], graph.nodes[v]['pos']
                ax_cluster.plot([p1[0], p2[0]], [p1[1], p2[1]], color=AGENT_COLOURS[i], alpha=0.3, linewidth=0.8, zorder=2)

            # Draw Nodes (Colored by Cluster)
            if len(assignments) == len(node_ids) and len(node_ids) > 0:
                for k, n_id in enumerate(node_ids):
                    npos   = graph.nodes[n_id]['pos']
                    ntype  = graph.nodes[n_id]['type']
                    c_id   = int(assignments[k])
                    colour = cluster_colours(c_id / 9.0)
                    alpha  = 1.0 if c_id == cur_cluster else 0.45

                    if ntype == NodeType.FRONTIER:
                        ax_cluster.scatter(npos[0], npos[1], marker=FRONTIER_MARK, color=colour, s=70, zorder=4, alpha=alpha, edgecolors='white' if c_id == cur_cluster else 'none', linewidths=0.6)
                    else:
                        ax_cluster.scatter(npos[0], npos[1], marker=BREADCRUMB_MARK, color=colour, s=20, zorder=3, alpha=alpha * 0.8)
            else:
                for n_id in node_ids:
                    npos = graph.nodes[n_id]['pos']
                    ax_cluster.scatter(npos[0], npos[1], color=AGENT_COLOURS[i], s=20, zorder=3, alpha=0.7)

            # Draw Centroid and Target Radius (Colored by Agent)
            if centroid is not None:
                circle = plt.Circle(centroid, radius=12, color=AGENT_COLOURS[i], fill=False, linestyle='--', linewidth=1.4, alpha=0.7, zorder=5)
                ax_cluster.add_patch(circle)
                ax_cluster.scatter(*centroid, color=AGENT_COLOURS[i], s=80, zorder=6, marker='+', linewidths=1.5)

            # Draw Agent Position
            ax_cluster.scatter(*pos_agent, color=AGENT_COLOURS[i], s=120, zorder=7, edgecolors='white', linewidths=1.2, marker='D')

        style_ax(ax_cluster, 'Combined Cluster Assignments')

        fig.suptitle(f'Multi-Agent Exploration — Step {snap["step"] + 1}', color='white', fontsize=18, fontweight='bold', y=0.97)

    # ── Render and Save ──────────────────────────────────────────────────────
    ani = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=60, blit=False)

    try:
        print(f"  Attempting to save MP4 to {save_path}...")
        ani.save(save_path, writer='ffmpeg', fps=15)
        print("  MP4 video saved successfully!")
    except Exception as e:
        print(f"  Could not save MP4. Falling back to GIF format...")
        try:
            gif_path = save_path.replace('.mp4', '.gif')
            ani.save(gif_path, writer='pillow', fps=15)
            print(f"  GIF animation saved successfully to {gif_path}!")
        except Exception as e2:
            print(f"  Failed to save animation entirely: {e2}")

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
    config['max_steps'] = MAX_EVAL_STEPS 

    from environment.graph_environment import GraphEnvironment
    from agents.agent import Agent
    from policies.high_level.gat_actor_critic import CentralCritic

    # Initialize environment with config passed appropriately
    env = GraphEnvironment(config, num_agents=config['num_agents'], grid_width=config['grid_width'], grid_height=config['grid_height'], risk_threshold=config['risk_threshold'])
    
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
    
    # Save video with the cleaner 3-plot grid
    video_filename = f'episode_viz_ep{ep_num}.mp4'
    animate_episode(snapshots, save_path=video_filename)

if __name__ == '__main__':
    main()