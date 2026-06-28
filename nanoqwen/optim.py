from __future__ import annotations

import torch


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step_t: torch.Tensor,
    lr_t: torch.Tensor,
    beta1_t: torch.Tensor,
    beta2_t: torch.Tensor,
    eps_t: torch.Tensor,
    wd_t: torch.Tensor,
) -> None:
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t**step_t
    bias2 = 1 - beta2_t**step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)


def muon_step_fused(
    stacked_grads: torch.Tensor,
    stacked_params: torch.Tensor,
    momentum_buffer: torch.Tensor,
    second_momentum_buffer: torch.Tensor,
    momentum_t: torch.Tensor,
    lr_t: torch.Tensor,
    wd_t: torch.Tensor,
    beta2_t: torch.Tensor,
    ns_steps: int,
    red_dim: int,
) -> None:
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    orthogonalize_dtype = torch.bfloat16 if g.device.type == "cuda" else torch.float32
    x = g.to(orthogonalize_dtype)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_matrix = x.mT @ x
            b_matrix = b * a_matrix + c * (a_matrix @ a_matrix)
            x = a * x + x @ b_matrix
    else:
        for a, b, c in POLAR_EXPRESS_COEFFS[:ns_steps]:
            a_matrix = x @ x.mT
            b_matrix = b * a_matrix + c * (a_matrix @ a_matrix)
            x = a * x + b_matrix @ x
    g = x

    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    """Single-device Muon for matrix params plus AdamW for the rest."""

    def __init__(self, param_groups, compile_steps: bool = False):
        super().__init__(param_groups, defaults={})
        self.compile_steps = compile_steps
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        compiler_kwargs = {"dynamic": False, "fullgraph": True}
        if compile_steps:
            self._adamw_step_impl = torch.compile(adamw_step_fused, **compiler_kwargs)
            self._muon_step_impl = torch.compile(muon_step_fused, **compiler_kwargs)
        else:
            self._adamw_step_impl = adamw_step_fused
            self._muon_step_impl = muon_step_fused

    def _scalar_tensors(self, device: torch.device, values: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if self.compile_steps:
            return values
        return tuple(torch.tensor(value.item(), dtype=torch.float32, device=device) for value in values)

    def _step_adamw(self, group: dict) -> None:
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t = self._scalar_tensors(
                p.device,
                (
                    self._adamw_step_t,
                    self._adamw_lr_t,
                    self._adamw_beta1_t,
                    self._adamw_beta2_t,
                    self._adamw_eps_t,
                    self._adamw_wd_t,
                ),
            )
            self._adamw_step_impl(p, grad, state["exp_avg"], state["exp_avg_sq"], step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t)

    def _step_muon(self, group: dict) -> None:
        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state or state["momentum_buffer"].shape[0] != num_params:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state or state["second_momentum_buffer"].shape[0] != num_params:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([param.grad for param in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        momentum_t, lr_t, wd_t, beta2_t = self._scalar_tensors(
            device,
            (self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t),
        )
        self._muon_step_impl(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            momentum_t,
            lr_t,
            wd_t,
            beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)
        return loss
