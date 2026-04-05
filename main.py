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
from policies.high_level.gat_actor_critic import GATActorCritic, CentralCritic

def main():
    # 1. Configuration & Setup
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    is_training = config.get('train_mode', True)
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    env = GraphEnvironment(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_policy = GATActorCritic(
        input_dim=7,
        hidden_dim=config.get('hidden_dim', 64),
        num_clusters=config.get('num_clusters', 4)
    ).to(device)
    
    agents = [Agent(i, config, shared_policy) for i in range(config['num_agents'])]
    
    central_critic = CentralCritic(
        hidden_dim=config.get('hidden_dim', 64),
        num_agents=config['num_agents'],
        num_clusters=config.get('num_clusters', 4)
    ).to(device) 

    learner = CTDELearner(agents, central_critic, shared_policy, lr=config.get('lr', 1e-4))

    latest = glob.glob(f"{checkpoint_dir}/shared_policy_ep*.pth")
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
        
        for agent in agents:
            # We don't care about catching the 3 variables for step 0 warmup
            agent.act(step=0, train=False, claimed_cluster_ids=set())

        # --- REMOVED: The old synchronized memory dict ---
        # memory = {
        #     'log_probs': [[] for _ in range(config['num_agents'])],
        #     'embeddings': [],   
        #     'rewards': [],
        #     'masks': []
        # }

        # --- ADDED: Independent Memory Architecture ---
        # 1. The Permanent Storage (The Batch Data)
        memory = {
            i: {
                'log_probs': [], 
                'global_embeddings': [], 
                'rewards': [],
                'durations': []
            } for i in range(config['num_agents'])
        }

        # 2. The "Shopping Carts" (Active Journeys)
        active_macro = {
            i: {
                'log_prob': None, 
                'reward': 0.0, 
                'duration': 0
            } for i in range(config['num_agents'])
        }

        for step in range(config.get('max_steps', 200)):
            # A. Update Perception (Decentralized)
            for i, agent in enumerate(agents):
                if step > 0: 
                    agent.update_perception(obs_buffers[i], step)

            # B. Coordination Phase (Neural Broadcasts)
            summaries = [a.get_broadcast_summary() for a in agents]
            for agent in agents:
                agent.belief.reset_broadcast_masks()
                peer_summaries = [s for s in summaries if s is None or s['agent_id'] != agent.agent_id]
                agent.belief.update_broadcast_masks(peer_summaries)

            # C. Strategic Action
            actions = {}

            for i, agent in enumerate(agents):
                if is_training:
                    # CHANGED: Catch the new 3-variable signature ---
                    target_pos, log_prob, is_decision = agent.act(
                        step, train=True, claimed_cluster_ids=set()
                    )
                    
                    # ADDED: The SMDP Gatekeeper Logic ---
                    if is_decision:
                        # 1. Checkout the old cart (if this isn't the very first step)
                        if active_macro[i]['log_prob'] is not None:
                            memory[i]['log_probs'].append(active_macro[i]['log_prob'])
                            memory[i]['rewards'].append(active_macro[i]['reward'])
                            memory[i]['durations'].append(active_macro[i]['duration'])
                        
                        # 2. Start a new cart
                        active_macro[i]['log_prob'] = log_prob
                        active_macro[i]['reward'] = 0.0
                        active_macro[i]['duration'] = 0
                        
                        # 3. Snapshot the global state (The X input)
                        # We grab ALL agents' embeddings at this exact millisecond.
                        step_embeddings = [a.last_embedding for a in agents]
                        global_state = torch.cat(step_embeddings, dim=-1) # [1, 1024]
                        memory[i]['global_embeddings'].append(global_state)
                else:
                    # CHANGED: Catch the new 3-variable signature ---
                    target_pos, _, _ = agent.act(
                        step, train=False, claimed_cluster_ids=set()
                    )

                actions[i] = target_pos

            # REMOVED: Store all agents' embeddings once per step ---
            # if is_training:
            #     step_embeddings = [a.last_embedding for a in agents]
            #     memory['embeddings'].append(step_embeddings)

            # D. Environment Step
            obs_buffers, team_reward, done, _ = env.step(actions)
            
            episode_reward += team_reward
            
            # CHANGED: Accumulate MACRO-REWARDS instead of single-step rewards ---
            if is_training:
                for i in range(config['num_agents']):
                    # If the agent is actively executing a plan, add the reward to its cart
                    if active_macro[i]['log_prob'] is not None:
                        active_macro[i]['reward'] += float(team_reward)
                        active_macro[i]['duration'] += 1

                # REMOVED: Old 1-step reward and mask logic ---
                # memory['rewards'].append(float(team_reward))
                # memory['masks'].append(float(1.0 - float(done)))

            # E. Visualization
            # if ep % config.get('render_freq', 10) == 0:
            #     env.render(agents=agents)

            if done: break

        # ADDED: End of Episode Cleanup ---
        # Don't throw away the final journeys just because the clock ran out!
        if is_training:
            for i in range(config['num_agents']):
                if active_macro[i]['log_prob'] is not None and active_macro[i]['duration'] > 0:
                    memory[i]['log_probs'].append(active_macro[i]['log_prob'])
                    memory[i]['rewards'].append(active_macro[i]['reward'])
                    memory[i]['durations'].append(active_macro[i]['duration'])

        # 3. Policy Update (Centralized Training)
        if is_training:
            learner.update_policy(memory)
            reward_history.append(episode_reward)
            
            with open('reward_log.json', 'w') as f:
                json.dump(reward_history, f)
            
            print(f"Episode {ep} | Total Reward: {episode_reward:.2f}")
            
            # Save Checkpoints
            if ep % config.get('save_freq', 50) == 0:
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