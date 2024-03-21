

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


def compute_soft_watkins_target(q_t, a_t, r_t, discount_t, c_t, pi_t):
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


def compute_loss(q_t, qT_t, a_t, a_t1, r_t, pi_t1, mu_t1,
                 discount_t, arms, running_errors, is_weights,
                 alpha=3., lambda_=0.95, kappa=0.01, n=0.5, eps=1e-8):
    """
    Implementation of MEME agent Bootstrapping with online network (A1),
    Loss and priority normalization (B1), and cross mixture training (B2)

    Apply inverse of value rescaling before passing into compute_retrace_target()
    Then, apply value rescaling after getting target from compute_retrace_target()

    Args:
        q_t (T+1, B, action_dim): expected q values at time t
        q_t1 (T+1, B, action_dim): target q values at time t+1
        a_t (T, B): actions at time t
        a_t1 (T, B): actions at time t+1
        r_t (T, B): rewards at time t
        pi_t1 (T, B, action_dim): target model policy head probs at time t+1
        mu_t1 (T, B, action_dim): target model action probs at time t+1
        discount_t (T, B): discount factor
        alpha (float=2.): value not specified in paper so set to 3. for now
                          (higher value = faster learning, lower value = more stable learning)
        lambda_ (int=0.95): lambda constant for retrace loss
        kappa (int=0.01):
        n (int=0.5):
        c_kl (int=0.5): Max KL for policy distillation
        eps (int=1e-2): small value to add to mu for numerical stability

    """
    T, B, N, action_dim = qT_t.shape

    assert q_t.shape == (T+1, B, N, action_dim)
    assert qT_t.shape == (T, B, N, action_dim)
    assert a_t.shape == (T, B, N)
    assert a_t1.shape == (T, B, N)
    assert r_t.shape == (T, B, N)
    assert pi_t1.shape == (T, B, N, action_dim)
    assert mu_t1.shape == (T, B, N)
    assert discount_t.shape == (T, B, N)
    assert arms.shape == (B,)
    assert is_weights.shape == (B,)

    assert not torch.isnan(q_t).any()
    assert not torch.isnan(qT_t).any()
    assert not torch.isnan(a_t).any()
    assert not torch.isnan(a_t1).any()
    assert not torch.isnan(r_t).any()
    assert not torch.isnan(pi_t1).any()
    assert not torch.isnan(mu_t1).any()
    assert not torch.isnan(discount_t).any()
    assert not torch.isnan(arms).any()
    assert not torch.isnan(is_weights).any()

    with torch.no_grad():
        # compute cutting trace coefficients in retrace
        # from what I understand: c_t1 is a way to correct for off policy samples

        # retrace:
        # pi_a_t1 = get_index(pi_t1, a_t)
        # c_t1 = torch.minimum(torch.tensor(1.0), pi_a_t1 / (mu_t1 + eps)) * lambda_

        # soft watkins Q(lambda): except that not all values are expected value
        q_a_t1 = get_index(q_t[1:], a_t1)
        assert not torch.isnan(q_a_t1).any()
        indicator = (q_a_t1.unsqueeze(-1) >= q_t[1:] - kappa * torch.abs(q_t[1:])).float()
        assert not torch.isnan(indicator).any()
        c_t1 = lambda_ * (pi_t1 * indicator).sum(-1)
        assert not torch.isnan(c_t1).any()

        # get transformed targets
        target = compute_soft_watkins_target(q_t[1:], a_t1, r_t, discount_t, c_t1, pi_t1)
        assert not torch.isnan(target).any()

    # get expected q value of taking action a_t
    expected = get_index(q_t[:-1], a_t)
    expectedT = get_index(qT_t, a_t)

    td_error = target - expected
    assert not torch.isnan(expected).any()
    assert not torch.isnan(target).any()
    assert not torch.isnan(td_error).any()

    # trust region mask (A1)
    with torch.no_grad():
        diff = expected - expectedT
        # according to 𝜎 = max(𝜎running, 𝜎batch, 𝜖), assuming batch is std of current td_errors
        # sigma = max(running_error.std(), td_error.std().item(), 0.01)
        running_stds = torch.tensor([x.std() for x in running_errors], device=diff.device)
        assert not torch.isnan(running_stds).any()
        batch_stds = td_error.view(-1, N).std(dim=0)
        assert not torch.isnan(batch_stds).any()
        sigma = torch.maximum(torch.maximum(running_stds, batch_stds), torch.tensor(0.01))
        assert not torch.isnan(sigma).any()
        mask = (torch.abs(diff) > alpha * sigma.view(1, 1, N).repeat(T, B, 1)) & (torch.sign(diff) != expected - target)

    # update 𝜎running
    for i, x in enumerate(running_errors):
        x.update(td_error[:, :, i].squeeze().cpu().detach().numpy().flatten())

    # loss and priority normalization (B1)
    # (T, B, N) / (N,)
    assert not torch.isnan(td_error).any()
    td_error = td_error / sigma
    assert not torch.isnan(td_error).any()

    loss = 0.5 * (td_error ** 2)
    assert not torch.isnan(loss).any()
    loss = torch.where(mask, 0., loss)
    assert not torch.isnan(loss).any()

    # Cross mixture loss (B2)
    assert not torch.isnan(loss).any()
    loss = n * get_index(loss, arms.unsqueeze(0).repeat(T, 1)) + ((1-n) / N) * loss.sum(-1)
    assert not torch.isnan(loss).any()
    loss *= is_weights
    loss = loss.mean()
    assert not torch.isnan(loss).any()

    return loss, td_error.sum(-1).detach()


def compute_policy_loss(q_t, pi_t, piT_t, c_kl=0.5):
    """
    Policy Distillation (D). To combat policy churn according to MEME paper.
    """
    action_size = q_t.size(-1)
    q_onehot = F.one_hot(torch.argmax(q_t, dim=-1), num_classes=action_size)

    p_loss = -(q_onehot * torch.log(pi_t)).sum(-1)

    mask = (F.kl_div(piT_t, pi_t, reduction="none").sum(-1) > c_kl)
    p_loss = torch.where(mask, 0., p_loss)
    p_loss = p_loss.mean()

    return p_loss

