def grad_finite_diff(function, w, eps=1e-8):
    return (function(w + eps) - function(w))/eps