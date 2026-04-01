import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from environment.graph_environment import GraphEnvironment

def main():
    print("Loading configuration...")
    # Load your existing config file
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("config.yaml not found! Please ensure it is in the same directory.")
        return

    # Initialize the environment
    print("Initializing Graph Environment...")
    env = GraphEnvironment(config, seed=42)

    num_episodes = 3
    max_test_steps = 50 # Keep it short just to test the visualizer and mechanics

    for ep in range(num_episodes):
        print(f"\n=== Starting Episode {ep + 1} ===")
        
        # Reset the environment (this will trigger the new Random Border Spawning)
        obs_buffers = env.reset()
        episode_reward = 0
        
        # Print initial spawn positions
        for i in range(env.num_agents):
            spawn_pos = env.agent_positions[i]
            print(f"Agent {i} spawned at: [{int(spawn_pos[0])}, {int(spawn_pos[1])}]")

        for step in range(max_test_steps):
            actions = {}
            
            # Generate random "drunk walk" actions for each agent
            for i in range(env.num_agents):
                current_pos = env.agent_positions[i]
                
                # Pick a random target within a 10-pixel radius of their current position
                random_step = np.random.uniform(-10, 10, size=2)
                target_x = np.clip(current_pos[0] + random_step[0], 0, env.grid_width - 1)
                target_y = np.clip(current_pos[1] + random_step[1], 0, env.grid_height - 1)
                
                actions[i] = np.array([target_x, target_y])

            # Step the environment forward
            obs_buffers, team_reward, done, _ = env.step(actions)
            episode_reward += team_reward

            # Render the environment every 5 steps
            if step % 5 == 0:
                env.render()
                # Adding a tiny sleep makes the animation easier to watch
                time.sleep(0.05) 

            if done:
                break
                
        print(f"Episode {ep + 1} finished with Total Team Reward: {episode_reward:.2f}")

    print("\nTesting complete. Close the plot window to exit.")
    plt.show() # Keeps the final visualizer window open

if __name__ == "__main__":
    main()