import numpy as np
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

@dataclass
class _Coordinate:
    """座標(a + b = 1)"""
    x: float # 無次元
    y: float # 無次元


@dataclass
class _Velocity:
    u: float 
    v: float

    def __init__(self, u: float, v: float):
        self.u = u
        self.v = v

    @property
    def speed(self) -> float:
        """速度の大きさを計算するプロパティ"""
        speed = np.sqrt(self.u**2 + self.v**2)
        return speed

@dataclass
class _Property:
    idx: int
    time: float
    coord: _Coordinate
    velocity: _Velocity


class RestrictedThreeBodyProblem:
    mu: float
    dt: float
    iter_max: int
    properties: list[_Property]
    coord0: _Coordinate
    velocity0: _Velocity

    def __init__(self, mu: float, x0: float, y0: float, u0: float, v0: float, dt: float = 0.1, iter_max: int = 20000):
        self.mu = mu
        self.dt = dt
        self.iter_max = iter_max
        initial_property = _Property(idx=0, time=0,coord= _Coordinate(x0, y0), velocity=_Velocity(u0, v0))
        self.properties = [initial_property]

    def calc(self):
        ni = 1
        while True:
            for _ in range(20):
                prop = self.properties[-1]
                x1 = prop.coord.x + 0.5 * self.dt * prop.velocity.u
                y1 = prop.coord.y + 0.5 * self.dt * prop.velocity.v
                u1 = prop.velocity.u + 0.5 * self.dt * self.du_dt(prop.coord, prop.velocity)
                v1 = prop.velocity.v + 0.5 * self.dt * self.dv_dt(prop.coord, prop.velocity)
                mid_prop = _Property(idx=-1, time=-1, coord=_Coordinate(x1, y1), velocity=_Velocity(u1, v1))

                x = prop.coord.x + self.dt*u1
                y = prop.coord.y + self.dt*v1
                u = prop.velocity.u + self.dt*self.du_dt(mid_prop.coord, mid_prop.velocity)
                v = prop.velocity.v + self.dt*self.dv_dt(mid_prop.coord, mid_prop.velocity)

                t = prop.time + self.dt
                prop = _Property(idx=ni, time=t, coord=_Coordinate(x, y), velocity=_Velocity(u, v))
                ni += 1
                self.properties.append(prop)
            if ni > self.iter_max:
                break
        return

    def du_dt(self, coord: _Coordinate, velocity: _Velocity) -> float:
        return self.d2x_dt2(coord, velocity)
    
    def dv_dt(self, coord: _Coordinate, velocity: _Velocity) -> float:
        return self.d2y_dt2(coord, velocity)
    
    def d2x_dt2(self, coord: _Coordinate, velocity: _Velocity) -> float:
        return -(1 - self.mu) * (coord.x - self.mu) / self.r1(coord)**3 - self.mu * (coord.x + 1 - self.mu) / self.r2(coord)**3 + coord.x + 2*velocity.v

    def d2y_dt2(self, coord: _Coordinate, velocity: _Velocity) -> float:
        return -(1 - self.mu)*coord.y/self.r1(coord)**3 - self.mu*coord.y/self.r2(coord)**3 + coord.y - 2*velocity.u

    def r1(self, coord: _Coordinate) -> float:
        return np.sqrt((coord.x - self.mu)**2 + coord.y**2)

    def r2(self, coord: _Coordinate) -> float:
        return np.sqrt((coord.x + 1 - self.mu)**2 + coord.y**2)
    
    def to_dataframe(self) -> pd.DataFrame:
        data = {
            "idx": [prop.idx for prop in self.properties],
            "time": [prop.time for prop in self.properties],
            "x": [prop.coord.x for prop in self.properties],
            "y": [prop.coord.y for prop in self.properties],
            "u": [prop.velocity.u for prop in self.properties],
            "v": [prop.velocity.v for prop in self.properties]
        }
        return pd.DataFrame(data)
    
    def plot(self, title: str|None = None):
        df = self.to_dataframe()
        plt.plot(df["x"], df["y"])
        plt.plot([self.mu, self.mu - 1], [0, 0], 'o')
        circle = Circle((0, 0), radius=self.mu, fill=False)
        circle2 = Circle((0, 0), radius=(1-self.mu), fill=False)
        plt.gca().add_patch(circle)
        plt.gca().add_patch(circle2)
        plt.xlim(-1.8, 1.8)
        plt.ylim(-1.8, 1.8)
        if title:
            plt.title(title)
        plt.show()
    
    