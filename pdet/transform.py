import jax
import jax.numpy as jnp
from jaxtyping import Array


def _mass_ratio(m1: Array, m2: Array) -> Array:
    safe_m1_from_zero = jnp.where(m1 == 0, jnp.finfo(jnp.result_type(m1)).min, m1)
    safe_m1_from_inf = jnp.where(
        safe_m1_from_zero == jnp.inf,
        jnp.finfo(jnp.result_type(m1)).max,
        safe_m1_from_zero,
    )
    safe_m2_from_inf = jnp.where(m2 == jnp.inf, jnp.finfo(jnp.result_type(m2)).max, m2)
    return safe_m1_from_inf / safe_m2_from_inf


def mass_ratio(*, m1: Array, m2: Array) -> Array:
    return jax.jit(_mass_ratio, inline=True)(m1=m1, m2=m2)


def _eta_from_q(q: Array) -> Array:
    safe_q_from_neg = jnp.where(q < 0, 1, q)
    log_eta = jnp.where(
        q < 0, -jnp.inf, jnp.log(safe_q_from_neg) + 2.0 * jnp.log1p(safe_q_from_neg)
    )
    return jnp.exp(log_eta)


def eta_from_q(q: Array) -> Array:
    return jax.jit(_eta_from_q, inline=True)(q)
