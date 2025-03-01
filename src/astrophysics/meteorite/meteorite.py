from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
GRAV = 980.665 # cm/s^2

@dataclass
class _Coordinate:
    x: float
    y: float

@dataclass
class _Velocity:
    u: float
    v: float

    def __init__(self, u: float, v: float):
        self.u = u
        self.v = v

    @property
    def speed(self) -> float:
        return np.sqrt(self.u**2 + self.v**2)
    
class MeteoriteProperty:
    time: float
    coord: _Coordinate
    velocity: _Velocity
    mass: float
    
    k1: float # 流星の抗力係数
    k2: float # 流星の抗力係数
    tau: float  # 流星の発光効率

    def __init__(self, time: float, coord: _Coordinate, velocity: _Velocity, mass: float, k1, k2, tau):
        self.time = time
        self.coord = coord
        self.velocity = velocity
        self.mass = mass
        self.k1 = k1
        self.k2 = k2
        self.tau = tau

    @property
    def density_atmosphere(self) -> float:
        """大気密度を計算するプロパティ"""
        return np.exp(-6.65125 - 1.39813e-6 * self.coord.y)
    
    @property 
    def du_dt(self) -> float:
        """速度の変化を計算するプロパティ"""
        if self.coord.y <= 0 or self.mass <= 0:
            return 0
        return -self.k1 * (self.mass**(-1/3)) * self.density_atmosphere * self.velocity.speed * self.velocity.u
    
    @property
    def dv_dt(self) -> float:
        """速度の変化を計算するプロパティ"""
        if self.coord.y <= 0 or self.mass <= 0:
            return 0
        return -self.k1 * (self.mass**(-1/3)) * self.density_atmosphere * self.velocity.speed * self.velocity.v - GRAV
    
    @property
    def dmass_dt(self) -> float:
        """質量の変化を計算するプロパティ"""
        if self.coord.y <= 0 or self.mass <= 0:
            return 0
        return -self.k2 * self.mass**(2/3) * self.density_atmosphere * self.velocity.speed**3

    @property 
    def luminosity(self) -> float:
        """流星の光度を計算するプロパティ"""
        return 0.5 * self.tau * self.dmass_dt * self.velocity.speed**2
    
    @property
    def apparent_magnitude(self) -> float:
        """流星の等級を計算するプロパティ"""
        if self.coord.y <= 0 or self.luminosity <= 0:
            return 0
        
        return 5 * np.log10(self.coord.y) - 2.5 * np.log10(self.luminosity) - 8.795
    

class Meteorite:
    """流星の軌道と高度を計算するクラス"""
    properties: list[MeteoriteProperty]
    
    def __init__(self, y0: float, u: float, v: float, m0: float, k1: float, k2: float, tau: float):
        initial_property = MeteoriteProperty(0, _Coordinate(0, y0), _Velocity(u, v), m0, k1, k2, tau)
        self.properties = [initial_property]

    @property
    def dt(self) -> float:
        m0 = self.properties[0].mass
        m = self.properties[-1].mass
        if m > 0.8 * m0:
            return 0.1
        elif m > 0.5 * m0:
            return 0.05
        elif m > 0.35 * m0:
            return 0.02
        else:
            return 0.01
        
    @property
    def _is_continue(self) -> bool:
        
        m0 = self.properties[0].mass
        m = self.properties[-1].mass
        # print(m, m0, "is_continue", m > m0 * 0)
        return m > m0 * 0.01

    def calc(self):
        print("time\tx\ty\tu\tv\tm")
        while self._is_continue:
             
            meteo = self.properties[-1]
            print(f"{meteo.time:.1e}\t{meteo.coord.x:.1e}\t{meteo.coord.y:.1e}\t{meteo.velocity.u:.1e}\t{meteo.velocity.v:.1e}\t{meteo.mass:.1e}")
            t = meteo.time + self.dt
            x1 = self.dt * meteo.velocity.u
            y1 = self.dt * meteo.velocity.v
            u1 = self.dt * meteo.du_dt
            v1 = self.dt * meteo.dv_dt
            m1 = self.dt * meteo.dmass_dt

            # print(f"{t:.1e}\t{x1:.1e}\t{y1:.1e}\t{u1:.1e}\t{v1:.1e}\t{m1:.1e}"
            meteo_for_calc = MeteoriteProperty(t, _Coordinate(x1, y1), _Velocity(u1, v1), m1, meteo.k1, meteo.k2, meteo.tau)
            x = meteo.coord.x + 0.5 * self.dt * (meteo.velocity.u + meteo_for_calc.velocity.u)
            y = meteo.coord.y + 0.5 * self.dt * (meteo.velocity.v + meteo_for_calc.velocity.v)
            u = meteo.velocity.u + 0.5 * self.dt * (meteo.du_dt + meteo_for_calc.du_dt)
            v = meteo.velocity.v + 0.5 * self.dt * (meteo.dv_dt + meteo_for_calc.dv_dt)
            m = meteo.mass + 0.5 * self.dt * (meteo.dmass_dt + meteo_for_calc.dmass_dt)
            
            print(f"{t:.1e}\t{x:.1e}\t{y:.1e}\t{u:.1e}\t{v:.1e}\t{m:.1e}")
            next_meteo = MeteoriteProperty(t, _Coordinate(x, y), _Velocity(u, v), m, meteo.k1, meteo.k2, meteo.tau)
            self.properties.append(next_meteo)

            if y <= 0:
                break
            if len(self.properties) > 1000:
                print("Error: too many iterations")
                break

    def to_dataframe(self) -> pd.DataFrame:
        """Convert properties to pandas DataFrame"""
        data = {
            'time': [p.time for p in self.properties],
            'x': [p.coord.x for p in self.properties],
            'y': [p.coord.y for p in self.properties],
            'u': [p.velocity.u for p in self.properties],
            'v': [p.velocity.v for p in self.properties],
            'mass': [p.mass for p in self.properties],
            'density': [p.density_atmosphere for p in self.properties],
            'luminosity': [p.luminosity for p in self.properties],
            'magnitude': [p.apparent_magnitude for p in self.properties]
        }
        return pd.DataFrame(data)
    
    def plot(self):
        """Plot meteorite properties using matplotlib"""

        data = self.to_dataframe()
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        
        # Plot trajectory
        axs[0,0].plot(data['x'], data['y'])
        axs[0,0].set_xlabel('x (cm)')
        axs[0,0].set_ylabel('y (cm)')
        axs[0,0].set_title('Trajectory')
        
        # Plot velocities
        axs[0,1].plot(data['time'], data['u'], label='u')
        axs[0,1].plot(data['time'], data['v'], label='v')
        axs[0,1].set_xlabel('time (s)')
        axs[0,1].set_ylabel('velocity (cm/s)')
        axs[0,1].set_title('Velocities')
        axs[0,1].legend()
        
        # Plot mass
        axs[1,0].plot(data['time'], data['mass'])
        axs[1,0].set_xlabel('time (s)')
        axs[1,0].set_ylabel('mass (g)')
        axs[1,0].set_title('Mass')
        
        # Plot apparent magnitude
        axs[1,1].plot(data['time'], data['magnitude'])
        axs[1,1].set_xlabel('time (s)')
        axs[1,1].set_ylabel('magnitude')
        axs[1,1].set_title('Apparent Magnitude')
        

        plt.tight_layout()
        plt.show()
        return fig, axs

 
    


