import numpy as np
from enum import Enum


class CIRDiscretizationScheme: 

    def __init__(self):
        pass

    def step(self, v_t: np.ndarray, dt : float, kappa: float, theta: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError("CIRDiscretizationScheme is an abstract base class. Implement the step method in a subclass.")
    

class EulerDiscretizationConvention(Enum):
    ABSORPTION = "absorption"
    REFLECTION = "reflection"
    FULL_TRUNCATION = "full_truncation"


class CIRDiscretizationSchemeEuler(CIRDiscretizationScheme):
    def __init__(self, convention: EulerDiscretizationConvention = EulerDiscretizationConvention.FULL_TRUNCATION):
        self.convention = convention

    def step(self, v_t, dt, kappa, theta, xi, z_v):
        if self.convention == EulerDiscretizationConvention.ABSORPTION:
            v_t_floored = np.maximum(v_t, 0)
            v_next = v_t_floored + kappa * (theta - v_t_floored) * dt + xi * np.sqrt(v_t_floored * dt) * z_v
            v_next = np.maximum(v_next, 0)  # on force aussi le resultat a rester >= 0
            return v_next, v_t_floored

        if self.convention == EulerDiscretizationConvention.REFLECTION:
            v_t_floored = np.abs(v_t)
            v_next = v_t_floored + kappa * (theta - v_t_floored) * dt + xi * np.sqrt(v_t_floored * dt) * z_v
            v_next = np.abs(v_next)  # on reflechit aussi le resultat
            return v_next, v_t_floored

        if self.convention == EulerDiscretizationConvention.FULL_TRUNCATION:
            v_t_floored = np.maximum(v_t, 0)
            v_next = v_t + kappa * (theta - v_t_floored) * dt + xi * np.sqrt(v_t_floored * dt) * z_v
            # v_next reste libre, PAS de floor sur le resultat
            return v_next, v_t_floored

        raise NotImplementedError(f"Convention {self.convention} non gerée.")
    

