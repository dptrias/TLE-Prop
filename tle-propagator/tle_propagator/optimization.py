from collections.abc import Callable
import numpy as np
from dataclasses import dataclass
from time import time

@dataclass
class OptimizationResult:
    """
    Container for optimization results.
    """
    success: bool
    x_opt: np.ndarray
    fun_opt: float
    f_hist: np.ndarray
    elapsed: float

    def __str__(self) -> str:
        status = "Success" if self.success else "Failure"
        return (f"Optimization Result:\n"
                f"Status: {status}\n"
                f"Optimal Parameters: {self.x_opt}\n"
                f"Optimal Function Value: {self.fun_opt:.6e}\n"
                f"Elapsed Time: {self.elapsed:.4f} seconds\n"
                f"Average time per iteration: {self.elapsed / len(self.f_hist):.4f} seconds")

def minimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    verbose: bool = False,
    max_iter: int = 1000,
    eps: float = 1e-8,
    tol: float = 1e-6,
    alpha_0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.5
) -> 'OptimizationResult':
    """
    Minimize the objective function using Newton's method.

    Parameters
    ----------
    fun : Callable[[np.ndarray], float]
        The objective function to minimize.
    x0 : np.ndarray
        Initial guess for the parameters.
    verbose : bool
        If True, print progress messages.
    max_iter : int
        Maximum number of iterations.
    eps : float
        Step size for numerical gradient computation.
    tol : float
        Tolerance for convergence.
    alpha_0 : float
        Initial step size for line search.
    c1 : float
        Parameter for Armijo condition (sufficient decrease).
    c2 : float
        Reduction factor for backtracking line search.

    Returns
    -------
    OptimizationResult
        The result of the optimization.
    """
    f_hist = []

    x = x0.copy()
    x_best = x0.copy()
    f_best = fun(x0)
    f_hist.append(f_best)

    # Initial evaluation
    f_val = fun(x)
    grad = numerical_gradient(fun, x, eps)
    
    tic = time()
    for i in range(max_iter):
        # Newton direction
        # hess = numerical_hessian(fun, x, grad, eps)
        # try:
        #     p_k = np.linalg.solve(hess, -grad)
        # except np.linalg.LinAlgError:
        #     print("[optimizer] Hessian is singular, using negative gradient as descent direction.")
        #     p_k = -grad
        p_k = -grad

        line_success, x_new, f_new, alpha, grad_new = line_search(fun, x, f_val, grad, p_k, alpha_0, c1, c2, eps, verbose)

        if not line_success:
            if verbose:
                print("[optimizer] Line search failed, stopping optimization.")
            return OptimizationResult(
                success = False,
                x_opt = x_best,
                fun_opt = f_best,
                f_hist = np.array(f_hist),
                elapsed = time() - tic
            )

        f_hist.append(f_new)
        if verbose:
            print(f"[optimizer] Iter {i+1:03d}: f = {f_new:.6e}, ||grad|| = {np.linalg.norm(grad_new):.6e}, alpha = {alpha:.2e}, Deltax = {np.linalg.norm(x_new - x):.3e}")
        # Update best found solution
        if f_new < f_best:
            f_best = f_new
            x_best = x_new.copy()

        # Check convergence
        if np.linalg.norm(x - x_new) < tol:
            if verbose:
                print(f"[optimizer] Converged in {i+1} iterations.")
            return OptimizationResult(
                success = True,
                x_opt = x_best,
                fun_opt = f_best,
                f_hist = np.array(f_hist),
                elapsed = time() - tic
            )
        
        # Update state
        x = x_new
        f_val = f_new
        grad = grad_new

    if verbose:
        print("[optimizer] Maximum iterations reached without convergence.")
        
    return OptimizationResult(
        success=False,
        x_opt=x,
        fun_opt=f_val,
        f_hist=np.array(f_hist),
        elapsed = time() - tic
    )

def numerical_gradient(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    eps: float
) -> np.ndarray:
    """
    Compute the numerical gradient of the function at point x using central differences.
    Optimized to avoid unnecessary array copies.
    """
    grad = np.zeros_like(x)
    for i in range(len(x)):
        og_x = x[i]
        
        x[i] = og_x + eps
        f_plus = fun(x)
        
        x[i] = og_x - eps
        f_minus = fun(x)
        
        x[i] = og_x
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
        
    return grad

def numerical_hessian(
    fun: Callable[[np.ndarray], float],
    x: np.ndarray,
    grad: np.ndarray,
    eps: float
) -> np.ndarray:
    """
    Compute the numerical Hessian of the function at point x.
    """
    n = len(x)
    hessian = np.zeros((n, n))
    
    for i in range(n):
        original_val = x[i]
        x[i] = original_val + eps
        
        grad_eps = numerical_gradient(fun, x, eps)
        hessian[:, i] = (grad_eps - grad) / eps
        
        x[i] = original_val # Restore
        
    return hessian

def line_search(
    fun: Callable[[np.ndarray], float],
    x_k: np.ndarray,
    f_k: float,
    grad_k: np.ndarray,
    p_k: np.ndarray,
    alpha_0: float,
    c1: float,
    c2: float,
    eps: float,
    verbose: bool
) -> tuple[bool, np.ndarray, float, float, np.ndarray]:
    """
    Wolfe conditions line search.
    """
    # Descent direction
    descent = np.dot(grad_k, -p_k)

    alpha = alpha_0
    x_new, f_new, grad_new = x_k, f_k, grad_k

    for _ in range(20):
        x_new = x_k + alpha * p_k
        f_new = fun(x_new)

        # Upper limit of alpha
        if f_new - f_k > c1 * alpha * descent:
            alpha *= c2
            continue

        grad_new = numerical_gradient(fun, x_new, eps)
        descent_new = np.dot(grad_new, -p_k)

        # Lower limit of alpha
        if descent_new < c2 * descent:
            alpha /= c2
            continue

        # Step size out of bounds
        if alpha < 1e-12 or alpha > 1e12:
            alpha = 1e-12
            if verbose:
                print("[optimizer] Line search step size out of bounds.")
            break

        # Both Wolfe conditions satisfied
        return True, x_new, f_new, alpha, grad_new

    if verbose:
        print("[optimizer] Unable to find suitable step size, softening conditions.")

    c2 /= 10.0
    if c2 < c1:
        if verbose:
            print("[optimizer] Line search parameters c1 and c2 have become invalid. Stopping line search.")
        return False, x_new, f_new, alpha, grad_new
    else:
        return line_search(fun, x_k, f_k, grad_k, p_k, alpha_0, c1, c2, eps, verbose)
