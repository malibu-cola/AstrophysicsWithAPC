"""彗星について扱う"""

from dataclasses import dataclass
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


class CoordinateSystem(Enum):
    """座標系の列挙子"""

    SOLAR_SYSTEM = 0
    COMET_SYSTEM = 1


@dataclass
class _Coordinate:
    """座標を表すクラス"""

    system: CoordinateSystem
    x: float | np.ndarray
    y: float | np.ndarray

    def to_solar_system(self, comet_nuclear_coord: "_Coordinate") -> "_Coordinate":
        """座標系を太陽系に変換する"""
        match comet_nuclear_coord.system:
            case CoordinateSystem.SOLAR_SYSTEM:
                x = comet_nuclear_coord.x
                y = comet_nuclear_coord.y
                r = np.sqrt(x**2 + y**2)
            case CoordinateSystem.COMET_SYSTEM:
                raise ValueError("comet_nuclear_coord must be in COMET_SYSTEM")

        match self.system:
            case CoordinateSystem.SOLAR_SYSTEM:
                return self
            case CoordinateSystem.COMET_SYSTEM:
                # TODO
                s = self.x
                t = self.y

                x_prime = (x * r + x * s + t * y) / r
                y_prime = (y * r - x * t + y * s) / r
                return _Coordinate(CoordinateSystem.SOLAR_SYSTEM, x_prime, y_prime)

    def to_comet_system(self, comet_nuclear_coord: "_Coordinate") -> "_Coordinate":
        """座標系を彗星系に変換する"""
        match comet_nuclear_coord.system:
            case CoordinateSystem.SOLAR_SYSTEM:
                x = comet_nuclear_coord.x
                y = comet_nuclear_coord.y
                r = np.sqrt(x**2 + y**2)
            case CoordinateSystem.COMET_SYSTEM:
                raise ValueError("comet_nuclear_coord must be in COMET_SYSTEM")

        match self.system:
            case CoordinateSystem.SOLAR_SYSTEM:
                # TODO
                x_prime = self.x
                y_prime = self.y

                s = (x_prime * x + y_prime * y - r**2) / r
                t = (x_prime * y - x * y_prime) / r
                return _Coordinate(CoordinateSystem.COMET_SYSTEM, s, t)
            case CoordinateSystem.COMET_SYSTEM:
                return self


class Comet:
    """彗星の軌道を計算するクラス"""

    mu: float  # 力の比 mu = F_{rad} / F_{grav}
    eccentricity: float  # 離心率 e
    distance_of_perihelion: float  # 近日点距離 a_p

    def __init__(
        self, mu: float, eccentricity: float, distance_of_perihelion: float
    ) -> None:
        self.mu = mu
        self.eccentricity = eccentricity
        self.distance_of_perihelion = distance_of_perihelion

    @property
    def half_latus_rectum(self):
        """半直弦 p"""
        return self.distance_of_perihelion * (1 + self.eccentricity)

    @property
    def semi_major_axis(self):
        """長半径 a"""
        return self.distance_of_perihelion / (1 - self.eccentricity)

    @property
    def escape_velocity(self):
        """脱出速度 g [AU/58.132day]"""
        if 1 - self.mu > 10:
            return 0.20
        elif 4 <= 1 - self.mu <= 10:
            return 0.05
        elif 0 <= 1 - self.mu < 4:
            return 0.02
        else:
            raise ValueError("mu must be in the range [0, 1]")

    def radius_of_orbit(self, nu: float | np.ndarray) -> float | np.ndarray:
        """軌道半径 r"""
        return self.half_latus_rectum / (1 + self.eccentricity * np.cos(nu))

    def nuclear_coord(self, nu: float | np.ndarray) -> _Coordinate:
        """軌道上の座標(x, y)を計算する"""
        x = self.radius_of_orbit(nu) * np.cos(nu)
        y = self.radius_of_orbit(nu) * np.sin(nu)
        return _Coordinate(CoordinateSystem.SOLAR_SYSTEM, x, y)

    def draw(self):
        """彗星軌道と彗星のシンダイン曲線を描画する"""
        # 軌道を描画
        self._draw_orbit()

        # 適当な位置の彗星を指定
        # nu = np.random.default_rng().uniform(0, 2 * np.pi)
        nu = np.linspace(0, 2 * np.pi, 20)
        for nu in nu:
            self._draw_comet(nu)
            self._draw_syndyname(nu)
        plt.show()

    def _draw_orbit(self):
        """軌道を描画する"""
        nu = np.linspace(0, 2 * np.pi, 1000)
        # 太陽を描画
        plt.plot(0, 0, "o", color="red")

        # 彗星の軌道を描画
        nuclear_coord = self.nuclear_coord(nu)
        x = nuclear_coord.x
        y = nuclear_coord.y
        plt.plot(x, y)

        plt.xlim(-self.half_latus_rectum, self.half_latus_rectum)
        plt.ylim(-self.half_latus_rectum, self.half_latus_rectum)
        plt.gca().set_aspect("equal", adjustable="box")

    def _draw_comet(self, nu: float):
        """彗星を描画する"""
        nuclear_coord = self.nuclear_coord(nu)
        plt.plot(nuclear_coord.x, nuclear_coord.y, "o", color="blue")

    def calculate_syndyname(
        self, s: float | np.ndarray, escape_direction: float | np.ndarray
    ) -> float | np.ndarray:
        """シンダイン曲線の座標を計算する"""
        if (isinstance(s, float) and s < 0) or (
            isinstance(s, np.ndarray) and (s < 0).any()
        ):
            raise ValueError("s must be greater than 0. s = {}".format(s))
        return self.escape_velocity * np.sin(escape_direction) * (
            self._a1(s) * np.sqrt(s) - self._a2(s) * s
        ) + self._a3(s) * s ** (3 / 2)

    def _a1(self, nu: float | np.ndarray) -> float | np.ndarray:
        """シンダイン曲線の係数A1"""
        return (np.sqrt(2) * self.radius_of_orbit(nu)) / np.sqrt(1 - self.mu)

    def _a2(self, nu: float | np.ndarray) -> float | np.ndarray:
        """シンダイン曲線の係数A2"""
        return (4.0 * self.eccentricity * self.radius_of_orbit(nu) * np.sin(nu)) / (
            3 * (1 - self.mu) * np.sqrt(self.half_latus_rectum)
        )

    def _a3(self, _nu: float | np.ndarray) -> float | np.ndarray:
        """シンダイン曲線の係数A3"""
        return (2 * np.sqrt(2 * self.half_latus_rectum)) / (3 * np.sqrt(1 - self.mu))

    def _draw_syndyname(self, nu: float):
        """シンダイン曲線を描画する"""
        escape_directions = np.linspace(-np.pi, np.pi, 10)  # 太陽と脱出角度のなす角度G

        for escape_direction in escape_directions:
            particle_coord_s = np.linspace(
                0.05, 0.5, 100
            )  # 彗星座標系における粒子の座標s

            particle_coord_t = self.calculate_syndyname(
                particle_coord_s, escape_direction
            )  # 彗星座標系における粒子の座標t

            comet_particle_coord = _Coordinate(
                CoordinateSystem.COMET_SYSTEM, particle_coord_s, particle_coord_t
            )
            nuclear_coord = self.nuclear_coord(nu)

            comet_particle_coord = comet_particle_coord.to_solar_system(nuclear_coord)
            plt.plot(
                comet_particle_coord.x,
                comet_particle_coord.y,
                "o",
                color="blue",
                markersize=2,
            )

        plt.gca().set_aspect("equal", adjustable="box")
