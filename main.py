import yaml
import numpy as np
import os
import glob
import json
import torch
import matplotlib.pyplot as plt
from environment.graph_environment import GraphEnvironment
from agents.agent import Agent
from training.learner import IndependentLearner

def main():
    # 1. Configuration & Setup
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    is_training = config.get('train_mode', True)
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    env = GraphEnvironment(config)
    
    agents = [Agent(i, config) for i in range(config['num_agents'])]
    
    learner = IndependentLearner(agents, lr=config.get('lr', 1e-4))

    # Checkpoint loading so training can resume after a crash
    latest = glob.glob(f"{checkpoint_dir}/agent_0_ep*.pth")
    if latest:
        latest.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]))
        ep_num = int(latest[-1].split('ep')[1].split('.')[0])
        print(f"Resuming from episode {ep_num}")
        for i, agent in enumerate(agents):
            ckpt_path = f"{checkpoint_dir}/agent_{i}_ep{ep_num}.pth"
            if os.path.exists(ckpt_path):
                agent.policy.load_state_dict(torch.load(ckpt_path, map_location=agent.device))
        

    # Tracking Stats
    reward_history = []
    
    # 2. Training Loop
    num_episodes = config.get('episodes', 500) if is_training else 1
    
    for ep in range(num_episodes):
        obs_buffers = env.reset()
        for agent in agents:
            agent.reset()
        
        episode_reward = 0 # This tracks the TOTAL team reward for plotting

        # Warm-up
        for i, agent in enumerate(agents):
            agent.update_perception(obs_buffers[i], step=0)
        
        warmup_claimed = set()
        for agent in agents:
            agent.act(step=0, train=False, claimed_cluster_ids=warmup_claimed)
            if agent.current_cluster_id is not None:
                warmup_claimed.add(agent.current_cluster_id)

        # Initialize memory for this episode
        memory = {
            'log_probs': [[] for _ in range(config['num_agents'])],
            'embeddings': [],   
            'rewards': [[] for _ in range(config['num_agents'])],
            'masks': []
        }

        for step in range(config.get('max_steps', 200)):
            # A. Update Perception
            for i, agent in enumerate(agents):
                if step > 0: 
                    agent.update_perception(obs_buffers[i], step)

            # B. Coordination Phase
            summaries = [a.get_broadcast_summary() for a in agents]
            for agent in agents:
                agent.belief.reset_broadcast_masks()
                peer_summaries = [s for s in summaries if s is None or s['agent_id'] != agent.agent_id]
                agent.belief.update_broadcast_masks(peer_summaries)

            # C. Strategic Action
            actions = {}
            claimed_cluster_ids = set() 

            for i, agent in enumerate(agents):
                if is_training:
                    target_pos, log_prob = agent.act(
                        step, train=True, claimed_cluster_ids=claimed_cluster_ids
                    )
                    memory['log_probs'][i].append(log_prob)
                else:
                    target_pos = agent.act(
                        step, train=False, claimed_cluster_ids=claimed_cluster_ids
                    )

                if agent.current_cluster_id is not None:
                    claimed_cluster_ids.add(agent.current_cluster_id)

                actions[i] = target_pos

            if is_training:
                step_embeddings = [a.last_embedding for a in agents]
                memory['embeddings'].append(step_embeddings)

            # D. Environment Step
            obs_buffers, individual_rewards, done, _ = env.step(actions)
            
            # ✅ FIXED: Sum individual rewards to track overall team performance
            episode_reward += sum(individual_rewards.values())

            if is_training:
                for i in range(config['num_agents']):
                    memory['rewards'][i].append(float(individual_rewards[i]))
                memory['masks'].append(float(1.0 - float(done)))

            # E. Visualization
            # if ep % config.get('render_freq', 10) == 0:
            #     env.render(agents=agents)

            if done: break

        # 3. Policy Update (Decentralized Training)
        if is_training:
            learner.update_policy(memory)
            reward_history.append(episode_reward)
            
            with open('reward_log.json', 'w') as f:
                json.dump(reward_history, f)
            
            print(f"Episode {ep} | Total Team Reward: {episode_reward:.2f}")
            
            # Save Checkpoints
            if ep % config.get('save_freq', 50) == 0:
                for i, agent in enumerate(agents):
                    torch.save(agent.policy.state_dict(), f"{checkpoint_dir}/agent_{i}_ep{ep}.pth")
                
                # ❌ REMOVED: central_critic saving
                
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