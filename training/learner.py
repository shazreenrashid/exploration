import torch
import torch.nn.functional as F

class IndependentLearner:
    # ❌ REMOVED: central_critic from init
    def __init__(self, agents, lr=1e-4, gamma=0.99):
        self.agents = agents
        self.gamma = gamma
        self.device = next(agents[0].policy.parameters()).device
        
        # ✅ CHANGED: Just one combined optimizer for all agents' policies 
        # (which now inherently includes their local critic heads)
        self.optimizer = torch.optim.Adam(
            [p for a in agents for p in a.policy.parameters()],
            lr=lr
        )

    def update_policy(self, memory):
        """
        memory: {
            'log_probs':  list[list] — one list per agent, each entry a log_prob tensor
            'embeddings': list[list] — one list per step, each entry is [agent0_z, agent1_z, ...]
            'rewards':    list[list] — ✅ CHANGED: one list per agent, each entry a float reward
            'masks':      list — one scalar per step (0.0 if done, else 1.0)
        }
        """
        masks = torch.tensor(memory['masks'], dtype=torch.float, device=self.device)
        
        total_combined_loss = 0

        # ✅ NEW: We now process each agent completely independently
        for i, agent in enumerate(self.agents):
            
            # 1. Get this specific agent's history
            agent_rewards = torch.tensor(memory['rewards'][i], dtype=torch.float, device=self.device)
            if len(agent_rewards) == 0:
                continue
                
            # 2. Compute discounted returns for this agent
            returns = self._compute_returns(agent_rewards, masks)  # [T]

            # 3. Compute values using the agent's OWN critic
            values = []
            for step_embeddings in memory['embeddings']:
                # step_embeddings[i] is the z_flat for THIS agent at THIS step
                z_flat = step_embeddings[i] 
                
                # Use the new helper method we added to GATActorCritic
                value = agent.policy.evaluate_value(z_flat)  # [1, 1]
                values.append(value.squeeze())
            
            values = torch.stack(values)  # [T]

            # 4. Compute advantage for this agent
            advantage = returns - values.detach()

            # 5. Actor loss for this agent
            agent_log_probs = torch.stack(memory['log_probs'][i])  # [T]
            policy_loss = -(agent_log_probs * advantage).mean()

            # 6. Critic loss for this agent
            value_loss = F.mse_loss(values, returns)

            # Accumulate the total loss across the team
            total_combined_loss += (policy_loss + value_loss)

        # 7. Update all networks at once
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