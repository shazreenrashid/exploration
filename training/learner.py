import torch
import torch.nn.functional as F

class CTDELearner:
    def __init__(self, agents, central_critic, lr=1e-4, gamma=0.99):
        self.agents = agents
        self.gamma = gamma
        self.central_critic = central_critic
        
        # Detect device from the first agent's policy
        self.device = next(agents[0].policy.parameters()).device
        
        # Separate optimizers for actors and centralised critic
        self.actor_optimizer = torch.optim.Adam(
            [p for a in agents for p in a.policy.parameters()],
            lr=lr
        )
        self.critic_optimizer = torch.optim.Adam(
            central_critic.parameters(),
            lr=lr
        )

    def update_policy(self, memory):
        """
        memory: {
            'log_probs':  list[list] — one list per agent, each entry a log_prob tensor
            'embeddings': list[list] — one list per step, each entry is [agent0_emb, agent1_emb, ...]
            'rewards':    list — one scalar per step
            'masks':      list — one scalar per step (0.0 if done, else 1.0)
        }
        """
        # 1. Convert rewards and masks to tensors
        rewards = torch.tensor(memory['rewards'], dtype=torch.float, device=self.device)
        masks = torch.tensor(memory['masks'], dtype=torch.float, device=self.device)
        
        # 2. Compute discounted returns
        returns = self._compute_returns(rewards, masks)  # [T]

        # 3. Compute values from centralised critic for each step
        # Each step's embeddings are [agent0_z, agent1_z, ...] where each z is [1, hidden_dim]
        values = []
        for step_embeddings in memory['embeddings']:
            value = self.central_critic(step_embeddings)  # [1, 1]
            values.append(value.squeeze())
        values = torch.stack(values)  # [T]

        # 4. Compute advantage using the centralised value estimate
        advantage = returns - values.detach()

        # 5. Actor loss — all agents share the same advantage signal
        policy_loss = 0
        for i in range(len(self.agents)):
            agent_log_probs = torch.stack(memory['log_probs'][i])  # [T]
            policy_loss -= (agent_log_probs * advantage).mean()

        # 6. Critic loss
        value_loss = F.mse_loss(values, returns)

        # # 7. Update actors
        # self.actor_optimizer.zero_grad()
        # policy_loss.backward(retain_graph=True)  # retain_graph because value_loss reuses values
        # torch.nn.utils.clip_grad_norm_(
        #     [p for a in self.agents for p in a.policy.parameters()], 1.0
        # )
        # self.actor_optimizer.step()

        # # 8. Update centralised critic
        # self.critic_optimizer.zero_grad()
        # value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.central_critic.parameters(), 1.0)
        # self.critic_optimizer.step()
        # 7. Update actors and critic together
        total_loss = policy_loss + value_loss
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for a in self.agents for p in a.policy.parameters()], 1.0
        )
        torch.nn.utils.clip_grad_norm_(self.central_critic.parameters(), 1.0)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def _compute_returns(self, rewards, masks):
        """Computes discounted returns. Returns a tensor on the same device as input."""
        R = torch.zeros(1, device=self.device)
        returns = torch.zeros_like(rewards)
        
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns[step] = R
            
        return returns


