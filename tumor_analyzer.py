import numpy as np


class EllipsoidFitter:
    def __init__(self, data=None, scale_factors=None, center_offset=None):
        self.data = np.array(data) if data is not None else np.array([])
        self.scale_factors = scale_factors
        if center_offset is None:
            center_offset = [0, 0, 0]
        self.center_offset = np.array(center_offset)

        self.center = None
        self.axes = None
        self.radii = None


    def set_precomputed_parameters(self, center, axes, radii):
        self.center = np.array(center)
        self.axes = np.array(axes)
        self.radii = np.array(radii)

    @classmethod
    def from_precomputed_parameters(cls, center, axes, radii):
        instance = cls()
        instance.set_precomputed_parameters(center, axes, radii)
        return instance

    def get_random_point(self):
        phi = np.random.uniform(0, 2 * np.pi)
        costheta = np.random.uniform(-1, 1)
        u = np.random.uniform(0, 1)
        theta = np.arccos(costheta)
        r = u ** (1 / 3)
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        random_point = np.dot([x, y, z], self.axes.T * self.radii) + self.center
        return random_point

    def get_ellipsoid_equation(self):
        """
        Get the mathematical equation of the fitted ellipsoid.
        """
        inv_radii_squared = 1.0 / (self.radii ** 2)
        equation = "Ellipsoid equation:\n"
        for i in range(3):
            term = f"(({self.axes[i, 0]} * (x - {self.center[0]}) + " \
                   f"{self.axes[i, 1]} * (y - {self.center[1]}) + " \
                   f"{self.axes[i, 2]} * (z - {self.center[2]}))^2) / " \
                   f"{1.0 / inv_radii_squared[i]}"
            if i < 2:
                term += " + "
            else:
                term += " = 1"
            equation += term + "\n"
        return equation

