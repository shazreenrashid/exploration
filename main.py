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
            # CHANGE: agent.act() now always returns (pos, log_prob, is_decision_step).
            # agent.act(step=0, train=False, claimed_cluster_ids=warmup_claimed)
            _, _, _ = agent.act(step=0, train=False, claimed_cluster_ids=warmup_claimed)
            if agent.current_cluster_id is not None:
                warmup_claimed.add(agent.current_cluster_id)

        # CHANGE: Independent Memory & Active Macro Tracker
        # memory = {
        #     'log_probs': [[] for _ in range(config['num_agents'])],
        #     'embeddings': [],   
        #     'rewards': [[] for _ in range(config['num_agents'])],
        #     'masks': []
        # }
        memory = {
            i: {'log_probs': [], 'embeddings': [], 'rewards': []} 
            for i in range(config['num_agents'])
        }
        active_macro = {
            i: {'log_prob': None, 'embedding': None, 'reward': 0.0} 
            for i in range(config['num_agents'])
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
                # CHANGE
                # if is_training:
                #     target_pos, log_prob = agent.act(
                #         step, train=True, claimed_cluster_ids=claimed_cluster_ids
                #     )
                #     memory['log_probs'][i].append(log_prob)
                # else:
                #     target_pos = agent.act(
                #         step, train=False, claimed_cluster_ids=claimed_cluster_ids
                #     )
                target_pos, log_prob, is_decision = agent.act(
                    step, train=is_training, claimed_cluster_ids=claimed_cluster_ids
                )
                actions[i] = target_pos

                if agent.current_cluster_id is not None:
                    claimed_cluster_ids.add(agent.current_cluster_id)

                # Why: If the agent made a NEW high-level choice, the old journey is over.
                if is_training and is_decision:
                    # 1. Save the previous completed journey to permanent memory
                    if active_macro[i]['log_prob'] is not None:
                        memory[i]['log_probs'].append(active_macro[i]['log_prob'])
                        memory[i]['embeddings'].append(active_macro[i]['embedding'])
                        memory[i]['rewards'].append(active_macro[i]['reward'])
                    
                    # 2. Put the NEW decision in the shopping cart
                    active_macro[i]['log_prob'] = log_prob
                    active_macro[i]['embedding'] = agent.last_embedding
                    active_macro[i]['reward'] = 0.0

            # CHANGE
            # if is_training:
            #     step_embeddings = [a.last_embedding for a in agents]
            #     memory['embeddings'].append(step_embeddings)

            # D. Environment Step
            obs_buffers, individual_rewards, done, _ = env.step(actions)
            
            # Sum individual rewards to track overall team performance
            episode_reward += sum(individual_rewards.values())

            # CHANGE
            # if is_training:
            #     for i in range(config['num_agents']):
            #         memory['rewards'][i].append(float(individual_rewards[i]))
            #     memory['masks'].append(float(1.0 - float(done)))
            if is_training:
                for i in range(config['num_agents']):
                    active_macro[i]['reward'] += float(individual_rewards[i])

            # E. Visualization
            # if ep % config.get('render_freq', 10) == 0:
            #     env.render(agents=agents)

            if done: break
            
        # CHANGE: If the episode timer runs out while agents are mid-journey, we need to save what they have so we don't lose the data.
        if is_training:
            for i in range(config['num_agents']):
                if active_macro[i]['log_prob'] is not None:
                    memory[i]['log_probs'].append(active_macro[i]['log_prob'])
                    memory[i]['embeddings'].append(active_macro[i]['embedding'])
                    memory[i]['rewards'].append(active_macro[i]['reward'])

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