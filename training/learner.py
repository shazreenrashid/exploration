import torch
import torch.nn.functional as F

class IndependentLearner:
    def __init__(self, agents, lr=1e-4, gamma=0.99):
        self.agents = agents
        self.gamma = gamma
        self.device = next(agents[0].policy.parameters()).device
        
        # Just one combined optimizer for all agents' policies 
        # (which now inherently includes their local critic heads)
        self.optimizer = torch.optim.Adam(
            [p for a in agents for p in a.policy.parameters()],
            lr=lr
        )

    def update_policy(self, memory):
        total_combined_loss = 0
        # We now process each agent completely independently
        for i, agent in enumerate(self.agents):
            # Pull this specific agent's isolated memory
            agent_memory = memory[i]
            # 1. Get this specific agent's history
            agent_rewards = torch.tensor(agent_memory['rewards'], dtype=torch.float, device=self.device)
            if len(agent_rewards) == 0:
                continue
                
            # 1.0 for all steps (keep future), 0.0 for the final step (no future left)
            masks = torch.ones_like(agent_rewards)
            masks[-1] = 0.0

            # 2. Compute discounted returns for this agent using the dynamic mask
            returns = self._compute_returns(agent_rewards, masks)  # [T]

            # 3. Compute values using the agent's OWN critic
            values = []
            for z_flat in agent_memory['embeddings']:
                # Evaluate the cached 256-dim embedding directly
                value = agent.policy.evaluate_value(z_flat)  # [1, 1]
                values.append(value.squeeze())
            
            values = torch.stack(values)  # [T]

            # 4. Compute advantage for this agent
            advantage = returns - values.detach()

            # 5. Actor loss for this agent
            agent_log_probs = torch.stack(agent_memory['log_probs'])  # [T]
            
            # Create a mathematical distribution out of our saved log_probs
            m = torch.distributions.Categorical(logits=agent_log_probs)

            # Calculate how "flat" or "random" the distribution is
            entropy = m.entropy().mean()
            
            # Add it to the loss! (We subtract because PyTorch tries to MINIMIZE loss)
            policy_loss = -(agent_log_probs * advantage).mean() - (0.01 * entropy)

            # 6. Critic loss for this agent
            # --- NEW: Scale value loss by 0.5 ---
            value_loss = 0.5 * F.mse_loss(values, returns)

            # Accumulate the total loss across the team
            total_combined_loss += (policy_loss + value_loss)

        # 7. Update all networks at once
        # Ensure we actually have a loss tensor to backpropagate before calling it
        if isinstance(total_combined_loss, torch.Tensor):
            self.optimizer.zero_grad()
            total_combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for a in self.agents for p in a.policy.parameters()], 1.0
            )
            self.optimizer.step()

    def _compute_returns(self, rewards, masks):
        R = torch.zeros(1, device=self.device)
        returns = torch.zeros_like(rewards)
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns[step] = R
        return returns