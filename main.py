import yaml
import numpy as np
import os
import glob
import json
import torch
import matplotlib.pyplot as plt
from environment.graph_environment import GraphEnvironment
from agents.agent import Agent
from training.learner import CTDELearner
# --- ADDED: We now need to import GATActorCritic here to create the shared brain ---
from policies.high_level.gat_actor_critic import GATActorCritic, CentralCritic

def main():
    # 1. Configuration & Setup
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    is_training = config.get('train_mode', True)
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    env = GraphEnvironment(
        config
    )

    # --- ADDED: Parameter Sharing Setup ---
    # WHAT: We instantiate a single GATActorCritic before creating the agents.
    # WHY: All agents will now use this exact same network in memory.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_policy = GATActorCritic(
        input_dim=7,
        hidden_dim=config.get('hidden_dim', 64),
        num_clusters=config.get('num_clusters', 4)
    ).to(device)
    
    # --- REMOVED: Old independent agent creation ---
    # agents = [Agent(i, config) for i in range(config['num_agents'])]
    
    # --- ADDED: Pass the shared_policy to every agent ---
    agents = [Agent(i, config, shared_policy) for i in range(config['num_agents'])]
    
    central_critic = CentralCritic(
        hidden_dim=config.get('hidden_dim', 64),
        num_agents=config['num_agents'],
        num_clusters=config.get('num_clusters', 4)
    ).to(device) # Can use device here safely

    # --- REMOVED: Old learner init ---
    # learner = CTDELearner(agents, central_critic, lr=config.get('lr', 1e-4))
    
    # --- ADDED: Pass shared_policy to the learner so it can optimize the shared weights ---
    learner = CTDELearner(agents, central_critic, shared_policy, lr=config.get('lr', 1e-4))

    # --- REMOVED: Old independent checkpoint loading ---
    # latest = sorted(glob.glob(f"{checkpoint_dir}/agent_0_ep*.pth"))
    # if latest:
    #     ep_num = int(latest[-1].split('ep')[1].split('.')[0])
    #     print(f"Resuming from episode {ep_num}")
    #     for i, agent in enumerate(agents):
    #         ckpt_path = f"{checkpoint_dir}/agent_{i}_ep{ep_num}.pth"
    #         if os.path.exists(ckpt_path):
    #             agent.policy.load_state_dict(torch.load(ckpt_path, map_location=agent.device))
                
    # --- ADDED: Shared policy checkpoint loading ---
    # WHAT: Load just the single shared_policy weights.
    # WHY: We only save one model now, no need to loop through agents.
    #latest = sorted(glob.glob(f"{checkpoint_dir}/shared_policy_ep*.pth"))
    latest = glob.glob(f"{checkpoint_dir}/shared_policy_ep*.pth")
    # Sort mathematically by the integer episode number
    latest.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]))
    if latest:
        ep_num = int(latest[-1].split('ep')[1].split('.')[0])
        print(f"Resuming from episode {ep_num}")
        
        ckpt_path = f"{checkpoint_dir}/shared_policy_ep{ep_num}.pth"
        if os.path.exists(ckpt_path):
            shared_policy.load_state_dict(torch.load(ckpt_path, map_location=device))
            
        critic_path = f"{checkpoint_dir}/central_critic_ep{ep_num}.pth"
        if os.path.exists(critic_path):
            central_critic.load_state_dict(torch.load(critic_path, map_location=device))

    # Tracking Stats
    reward_history = []
    
    # 2. Training Loop
    num_episodes = config.get('episodes', 500) if is_training else 1
    
    for ep in range(num_episodes):
        obs_buffers = env.reset()
        for agent in agents:
            agent.reset()
        episode_reward = 0

        for i, agent in enumerate(agents):
            agent.update_perception(obs_buffers[i], step=0)
        
        # --- REMOVED: The manual claims tracking for Step 0 ---
        # warmup_claimed = set()
        for agent in agents:
            # --- ADDED: Just pass empty set/None, removing manual hardcoded claims ---
            agent.act(step=0, train=False, claimed_cluster_ids=set())
            # if agent.current_cluster_id is not None:
            #     warmup_claimed.add(agent.current_cluster_id)

        # Initialize memory for this episode
        memory = {
            'log_probs': [[] for _ in range(config['num_agents'])],
            'embeddings': [],   
            'rewards': [],
            'masks': []
        }

        for step in range(config.get('max_steps', 200)):
            # A. Update Perception (Decentralized)
            for i, agent in enumerate(agents):
                if step > 0: #
                    agent.update_perception(obs_buffers[i], step)

            # B. Coordination Phase (Neural Broadcasts - WE KEEP THIS!)
            summaries = [a.get_broadcast_summary() for a in agents]
            for agent in agents:
                agent.belief.reset_broadcast_masks()
                peer_summaries = [s for s in summaries if s is None or s['agent_id'] != agent.agent_id]
                agent.belief.update_broadcast_masks(peer_summaries)

            # C. Strategic Action
            actions = {}
            # --- REMOVED: The manual claims set tracking ---
            # claimed_cluster_ids = set()  

            for i, agent in enumerate(agents):
                if is_training:
                    # --- ADDED: Pass empty set to agent.act. Let RL handle deconfliction. ---
                    target_pos, log_prob = agent.act(
                        step, train=True, claimed_cluster_ids=set()
                    )
                    
                    # --- CRITICAL FIX: REMOVED .detach() from log_prob ---
                    # WHAT: Changed log_prob.detach() to just log_prob
                    # WHY: If you detach the log_prob here, the gradients cannot flow back 
                    # from the loss function into the Actor network. Training will silently fail.
                    memory['log_probs'][i].append(log_prob)
                else:
                    target_pos = agent.act(
                        step, train=False, claimed_cluster_ids=set()
                    )

                # --- REMOVED: Manual cluster blocking logic ---
                # if agent.current_cluster_id is not None:
                #     claimed_cluster_ids.add(agent.current_cluster_id)

                actions[i] = target_pos

            # Store all agents' embeddings once per step
            if is_training:
                step_embeddings = [a.last_embedding for a in agents]
                memory['embeddings'].append(step_embeddings)

            # D. Environment Step
            obs_buffers, team_reward, done, _ = env.step(actions)
            
            episode_reward += team_reward
            if is_training:
                memory['rewards'].append(float(team_reward))
                memory['masks'].append(float(1.0 - float(done)))

            # E. Visualization
            # if ep % config.get('render_freq', 10) == 0:
            #     env.render(agents=agents)

            if done: break

        # 3. Policy Update (Centralized Training)
        if is_training:
            learner.update_policy(memory)
            reward_history.append(episode_reward)
            
            with open('reward_log.json', 'w') as f:
                json.dump(reward_history, f)
            
            print(f"Episode {ep} | Total Reward: {episode_reward:.2f}")
            
            # Save Checkpoints
            if ep % config.get('save_freq', 50) == 0:
                # --- REMOVED: Saving independent agent policies ---
                # for i, agent in enumerate(agents):
                #     torch.save(agent.policy.state_dict(), f"{checkpoint_dir}/agent_{i}_ep{ep}.pth")
                
                # --- ADDED: Save only the single shared policy ---
                # WHAT: One file to rule them all.
                # WHY: Since all agents share weights, we only need to write one policy to disk.
                torch.save(shared_policy.state_dict(), f"{checkpoint_dir}/shared_policy_ep{ep}.pth")
                torch.save(central_critic.state_dict(), f"{checkpoint_dir}/central_critic_ep{ep}.pth")
                
                # Plot training curve
                plt.figure(figsize=(10, 5))
                plt.plot(reward_history)
                plt.title("Team Reward Training Curve")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.savefig("training_curve.png")
                plt.close()

    print("Task Complete.")

if __name__ == "__main__":
    main()