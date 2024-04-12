

import torch
import torch.nn.functional as F

from .value_rescale import rescale, inv_rescale


def get_index(x, idx):
    """
    Code modified from
    https://github.com/deepmind/trfl/blob/master/trfl/retrace_ops.py
    https://github.com/michaelnny/deep_rl_zoo/blob/main/deep_rl_zoo/nonlinear_bellman.py

    Returns x that correspond to index in idx

    Args:
        x (T, B, action_dim): tensor to be indexed
        idx (T, B): indices

    Returns:
        indexed (T, B): indexed tensor
    """
    action_dim = x.size(-1)
    idx_shape = idx.shape

    x = x.reshape(-1, action_dim)
    idx = idx.reshape(-1)

    indexed = x[torch.arange(x.size(0)), idx]
    indexed = indexed.view(*idx_shape)

    return indexed


def compute_target(q_t, a_t, r_t, discount_t, c_t, pi_t):
    """
    Compute target for Transformed Retrace Operators

    According to https://github.com/deepmind/trfl/blob/master/trfl/retrace_ops.py:
    New action-value estimates (target value 'T') must be expressable in this
    recurrent form:

        T(x_{t-1}, a_{t-1}) = r_t + γ[ 𝔼_π Q(x_t, .) - c_t Q(x_t, a_t) + c_t T(x_t, a_t) ]

    Munos' et al. Retrace:

        c_t = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).

    Hence:

        T_tm1 = r_t + γ * (exp_q_t - c_t * qa_t) + γ * c_t * T_t

    Define:

        current = r_t + γ * (exp_q_t - c_t * qa_t)

    Args:
        q_t (T, B, action_dim): target network q value at time t+1
        a_t (T, B): action index at time t+1
        r_t (T, B): rewards at time t
        discount_t (T, B): discount at time t
        c_t (T, B): importance weights at time t+1
        pi_t (T, B, action_dim): policy probabilities from online network at time t+1


    TODO:
        understand the math derivations
    """
    q_t = inv_rescale(q_t)

    exp_q_t = (pi_t * q_t).sum(axis=-1)
    q_a_t = get_index(q_t, a_t)

    current = r_t + discount_t * (exp_q_t - c_t * q_a_t)
    decay = discount_t * c_t

    g = q_a_t[-1]
    returns = []
    for t in reversed(range(q_a_t.size(0))):
        g = current[t] + decay[t] * g
        returns.insert(0, g)

    return rescale(torch.stack(returns, dim=0).detach())


def compute_soft_watkins_loss(q_t, qT_t, a_t, a_t1, r_t, pi_t1, discount_t, arms, is_weights, running_errors,
                              lambda_=0.95, kappa=0.01, alpha=3.0, n=0.5, tau=0.25):
    """
    Apply inverse of value rescaling before passing into compute_retrace_target()
    Then, apply value rescaling after getting target from compute_retrace_target()

    Args:
        q_t (T, B, action_dim): expected q values at time t
        q_t1 (T, B, action_dim): target q values at time t+1
        a_t (T, B): actions at time t
        a_t1 (T, B): actions at time t+1
        r_t (T, B): rewards at time t
        pi_t1 (T, B, action_dim): online model action probs at time t+1
        discount_t (T, B): discount factor
        lambda_ (int=0.95): lambda constant for retrace loss
        eps (int=1e-2): small value to add to mu for numerical stability

    """
    T, B, N, action_dim = qT_t.shape

    assert q_t.shape == (T+1, B, N, action_dim)
    assert qT_t.shape == (T, B, N, action_dim)
    assert a_t.shape == (T, B, N)
    assert a_t1.shape == (T, B, N)
    assert r_t.shape == (T, B, N)
    assert pi_t1.shape == (T, B, N, action_dim)
    assert discount_t.shape == (T, B, N)
    assert arms.shape == (T, B)
    assert is_weights.shape == (B,)

    pi_t1 = F.softmax(F.log_softmax(pi_t1, dim=-1) / tau, dim=-1)

    with torch.no_grad():
        q_a_t1 = get_index(q_t[1:], a_t1)
        indicator = (q_a_t1.unsqueeze(-1) >= q_t[1:] - kappa * torch.abs(q_t[1:])).float()
        c_t1 = lambda_ * (pi_t1 * indicator).sum(-1)

        # get transformed soft watkins targets
        target = compute_target(q_t[1:], a_t1, r_t, discount_t, c_t1, pi_t1)

    # get expected q value of taking action a_t
    expected = get_index(q_t[:-1], a_t)
    expectedT = get_index(qT_t, a_t)

    td_error = target - expected

    # trust region mask (A1)
    with torch.no_grad():
        diff = expected - expectedT
        running_stds = torch.tensor([x.std() for x in running_errors], device=diff.device)
        batch_stds = td_error.view(-1, N).std(dim=0)

        sigma = torch.maximum(torch.maximum(running_stds, batch_stds), torch.tensor(0.01))
        mask = (torch.abs(diff) > alpha * sigma.view(1, 1, N).repeat(T, B, 1)) & (torch.sign(diff) != expected - target)

    # update 𝜎running
    for i, x in enumerate(running_errors):
        x.update(td_error[:, :, i].squeeze().cpu().detach().numpy().flatten())

    td_error = td_error / sigma

    loss = td_error ** 2
    loss = torch.where(mask, torch.tensor(0.), loss)

    # weight loss according to prioritized experience replay and then take mean
    loss = n * get_index(loss, arms) + ((1-n) / N) * loss.sum(-1)
    loss *= is_weights
    loss = loss.mean()

    return loss, td_error.detach()


def compute_policy_loss(q_t, pi_t, piT_t, c_kl=0.5, eps=1e-4):
    """
    Policy Distillation (D). To combat policy churn according to MEME paper.
    """
    T, B, N, action_dim = q_t.shape

    assert q_t.shape == (T, B, N, action_dim)
    assert pi_t.shape == (T, B, N, action_dim)
    assert piT_t.shape == (T, B, N, action_dim)

    assert not torch.isnan(q_t).any(), q_t
    assert not torch.isnan(pi_t).any(), pi_t
    assert not torch.isnan(piT_t).any(), piT_t

    # G_eps
    policy = torch.argmax(q_t, dim=-1)
    policy = F.one_hot(policy, num_classes=action_dim).to(torch.float32)

    # cross entropy
    p_loss = F.cross_entropy(pi_t.view(-1, action_dim), policy.view(-1, action_dim),
                             reduction='none', label_smoothing=eps)
    p_loss = p_loss.view(T, B, N)

    # mask out violation at time step, then sum over a, t
    # mask = (F.kl_div(F.log_softmax(piT_t, dim=-1), F.log_softmax(pi_t, dim=-1), reduction="none", log_target=True
    #                  ).mean(-1) <= c_kl)
    # p_loss = torch.where(mask, p_loss, 0.)
    p_loss = p_loss.sum(0).mean()

    return p_loss


if __name__ == "__main__":
    T, B, action_dim = 3, 2, 4

    x = torch.ones((T, B, action_dim))
    y = torch.ones((T, B))

    loss = compute_soft_watkins_loss(x, x, y, y, y, x, x, y)

