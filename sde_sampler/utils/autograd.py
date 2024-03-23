import torch
import torch.autograd as autograd


def _compute_autograd(outputs, inputs, create_graph=False, retain_graph=False):
    return autograd.grad(
        outputs.sum(),
        inputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
    )


def _compute_autodiv(outputs, inputs, create_graph=False):
    assert outputs.shape == inputs.shape
    div = 0.0
    for i in range(outputs.shape[-1]):
        div += _compute_autograd(
            outputs[:, i], inputs, create_graph=create_graph, retain_graph=True
        )[0][:, i : i + 1]
    return div


def _estimate_autodiv(
    outputs, inputs, n_samples=1, noise_type="rademacher", create_graph=False
):
    # Hutchinson trace estimator
    # Adapted from https://github.com/yang-song/score_sde_pytorch
    assert outputs.shape == inputs.shape
    div = 0.0
    for _ in range(n_samples):
        if noise_type == "rademacher":
            noise = torch.randint_like(outputs, low=0, high=2).float() * 2 - 1.0
        elif noise_type == "gauss":
            noise = torch.randn_like(outputs)
        else:
            raise NotImplementedError(f"Undefined noise type {noise_type}.")
        grad = torch.autograd.grad(
            outputs, inputs, grad_outputs=noise, create_graph=create_graph
        )[0]
        div += (grad * noise).sum(dim=-1, keepdims=True)
    return div / n_samples


def compute_derivatives(
    fn, t, x, *args, compute_laplacian=False, create_graph=True, **kwargs
):
    t_requires_grad = t.requires_grad
    x_requires_grad = x.requires_grad
    laplacian = None
    with torch.set_grad_enabled(True):
        t.requires_grad_(True)
        x.requires_grad_(True)
        outputs = fn(t, x, *args, **kwargs)
        gradt, gradx = _compute_autograd(
            outputs, (t, x), create_graph=create_graph, retain_graph=True
        )
        if compute_laplacian:
            laplacian = _compute_autodiv(gradx, x, create_graph=create_graph)
    t.requires_grad_(t_requires_grad)
    x.requires_grad_(x_requires_grad)
    if not torch.is_grad_enabled():
        outputs = outputs.detach()
    return gradt, gradx, laplacian, outputs


def compute_gradx(fn, t, x, *args, create_graph=True, retain_graph=False, **kwargs):
    requires_grad = x.requires_grad
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        outputs = fn(t, x, *args, **kwargs)
        gradient = _compute_autograd(
            outputs, x, create_graph=create_graph, retain_graph=retain_graph
        )[0]
    x.requires_grad_(requires_grad)
    if not torch.is_grad_enabled():
        outputs = outputs.detach()
    return gradient, outputs


def compute_divx(
    fn, t, x, *args, create_graph=True, noise_type=None, n_samples=1, **kwargs
):
    requires_grad = x.requires_grad
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        outputs = fn(t, x, *args, **kwargs)
        if noise_type is None:
            div = _compute_autodiv(
                outputs,
                x,
                create_graph=create_graph,
            )
        else:
            div = _estimate_autodiv(
                outputs,
                x,
                n_samples=n_samples,
                noise_type=noise_type,
                create_graph=create_graph,
            )
    x.requires_grad_(requires_grad)
    if not torch.is_grad_enabled():
        outputs = outputs.detach()
    return div, outputs
