import torch
import torch.nn.functional as F

class CTDELearner:
    def __init__(self, agents, central_critic, shared_policy, lr=1e-4, gamma=0.99):
        self.agents = agents
        self.gamma = gamma
        self.central_critic = central_critic
        self.shared_policy = shared_policy
        self.device = next(shared_policy.parameters()).device
        
        self.actor_optimizer = torch.optim.Adam(self.shared_policy.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(central_critic.parameters(), lr=lr)

    def update_policy(self, memory):
        """
        memory: dict separated by agent_id
        {
            0: {
                'log_probs':  [tensor, tensor, ...],
                'global_embeddings': [tensor, tensor, ...], # The 1024-dim global state snapshot
                'rewards':    [float, float, ...],          # The accumulated macro-reward
                'durations':  [int, int, ...]               # How many ticks the action took
            },
            1: { ... }
        }
        """
        policy_loss = 0
        value_loss = 0
        total_steps_processed = 0

        for i in range(len(self.agents)):
            agent_memory = memory[i]
            
            # If the agent never made a decision (or memory is empty), skip it
            if not agent_memory['log_probs']:
                continue

            # 1. Convert lists to tensors
            rewards = torch.tensor(agent_memory['rewards'], dtype=torch.float, device=self.device)
            durations = torch.tensor(agent_memory['durations'], dtype=torch.float, device=self.device)
            
            # 2. Compute SMDP discounted returns using durations instead of a flat mask
            returns = self._compute_returns(rewards, durations)  # [T]

            # 3. Evaluate the Central Critic using the saved global snapshots
            global_states = torch.cat(agent_memory['global_embeddings'], dim=0) # [T, 1024]
            values = self.central_critic(global_states).squeeze(-1)             # [T]

            # 4. Compute advantage
            advantage = returns - values.detach()

            # 5. Actor Loss for this specific agent's timeline
            agent_log_probs = torch.stack(agent_memory['log_probs'])  # [T]
            
            policy_loss -= (agent_log_probs * advantage).sum()

            # 6. Critic Loss for this specific agent's timeline
            value_loss += F.mse_loss(values, returns, reduction='sum')
            
            total_steps_processed += len(rewards)

        # Prevent division by zero if an episode ended instantly
        if total_steps_processed == 0:
            return

        # Average the losses across all decisions made by the team this episode
        policy_loss = policy_loss / total_steps_processed
        value_loss = value_loss / total_steps_processed
        total_loss = policy_loss + value_loss

        # 7. Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.shared_policy.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.central_critic.parameters(), 1.0)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def _compute_returns(self, rewards, durations):
        """Computes SMDP discounted returns taking action duration into account."""
        R = torch.zeros(1, device=self.device)
        returns = torch.zeros_like(rewards)
        
        for step in reversed(range(len(rewards))):
            # SMDP Math: Discount the future value by gamma^duration
            # (e.g., if it took 5 steps, gamma becomes 0.99^5)
            # The final step naturally has R=0, so it handles the terminal state perfectly.
            R = rewards[step] + (self.gamma ** durations[step]) * R
            returns[step] = R
            
        return returns