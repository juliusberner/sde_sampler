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
    div = 0.0
    for i in range(outputs.shape[-1]):
        div += _compute_autograd(
            outputs[:, i], inputs, create_graph=create_graph, retain_graph=True
        )[0][:, i : i + 1]
    return div


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
    return gradient, outputs


def compute_divx(fn, t, x, *args, create_graph=True, **kwargs):
    requires_grad = x.requires_grad
    with torch.set_grad_enabled(True):
        x.requires_grad_(True)
        outputs = fn(t, x, *args, **kwargs)
        div = _compute_autodiv(
            outputs,
            x,
            create_graph=create_graph,
        )
    x.requires_grad_(requires_grad)
    return div, outputs
