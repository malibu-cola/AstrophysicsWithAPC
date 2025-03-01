from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
GRAV = 980.665 # cm/s^2

@dataclass
class _Coordinate:
    x: float # [cm]
    y: float # [cm]

@dataclass
class _Velocity:
    u: float #[cm/s]
    v: float #[cm/s]

    def __init__(self, u: float, v: float):
        self.u = u
        self.v = v

    @property
    def speed(self) -> float:
        """速度の大きさを計算するプロパティ [cm/s]"""
        speed = np.sqrt(self.u**2 + self.v**2)
        return speed
    
class MeteoriteProperty:
    time: float # [s]
    coord: _Coordinate 
    velocity: _Velocity
    mass: float # [g]
    
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
        # print(f"meteorite: time: {time}, x: {coord.x}, y: {coord.y}, u: {velocity.u}, v: {velocity.v}, mass: {mass}")

    @property
    def density_atmosphere(self) -> float:
        """大気密度を計算するプロパティ [g/cm^3]"""
        ret = np.exp(-6.65125 - 1.39813e-6 * self.coord.y)
        # print("atmosphere: ", ret)
        return ret
    
    @property 
    def du_dt(self) -> float:
        """速度の変化を計算するプロパティ"""
        ret = 0
        if self.coord.y <= 0 or self.mass <= 0:
            # print(f"y = {self.coord.y}, mass: {self.mass}, du_dt: {ret}")
            return 0
        # ret = -self.k1 * (self.mass**(-1/3)) * self.density_atmosphere * self.velocity.speed * self.velocity.u
        ret = -self.k1 * self.density_atmosphere * self.velocity.speed * self.velocity.u * np.exp(-1/3 * np.log(self.mass))
        # print("du_dt: ",ret)
        return ret
    
    @property
    def dv_dt(self) -> float:
        """速度の変化を計算するプロパティ"""
        ret = 0
        if self.coord.y <= 0 or self.mass <= 0:
            # print(f"y: {self.coord.y}, mass: {self.mass}, dv_dt: {ret} ")
            return ret
        # ret =  -self.k1 * (self.mass**(-1/3)) * self.density_atmosphere * self.velocity.speed * self.velocity.v - GRAV
        ret = -self.k1 * self.density_atmosphere * self.velocity.speed * self.velocity.v * np.exp(-1/3 * np.log(self.mass)) - GRAV
        # print("dv_dt: ",ret)
        return ret
    
    @property
    def dmass_dt(self) -> float:
        """質量の変化を計算するプロパティ"""
        if self.coord.y <= 0 or self.mass <= 0:
            # print(f"y: {self.coord.y}, mass: {self.mass}, dmass_dt: 0")
            return 0
        # ret = -self.k2 * self.mass**(2/3) * self.density_atmosphere * self.velocity.speed**3
        ret = -self.k2 * self.density_atmosphere * (self.velocity.speed ** 3) * np.exp(2/3 * np.log(self.mass))
        # print("dmass_dt: ",ret)
        return ret

    @property 
    def luminosity(self) -> float:
        """流星の光度を計算するプロパティ"""
        luminosity =- 0.5 * self.tau * self.dmass_dt * (self.velocity.speed**2)
        # print(f"luminosity: {luminosity}")
        return luminosity
    
    @property
    def apparent_magnitude(self) -> float:
        """流星の等級を計算するプロパティ"""
        if self.coord.y <= 0 or self.luminosity <= 0:
            # print(f"y: {self.coord.y}, luminosity: {self.luminosity}, magnitude: 0")
            return 0
        magnitude = 5 * np.log10(self.coord.y) - 2.5 * np.log10(self.luminosity) - 8.795
        # print(f"magnitude: {magnitude}")
        return magnitude
    

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
        return m > m0 * 0.01

    def calc(self):
        # print("time\tx\ty\tu\tv\tm\trho\tspeed\tdudt\tdvdt\tdmassdt\tluminosity\tmagnitude")
        while self._is_continue:
             
            meteo = self.properties[-1]
            # print()
            # print(f"{meteo.time:.1e}\t{meteo.coord.x:.1e}\t{meteo.coord.y:.1e}\t{meteo.velocity.u:.1e}\t{meteo.velocity.v:.1e}\t{meteo.mass:.1e}")
            t = meteo.time + self.dt
            x1 = meteo.coord.x + self.dt * meteo.velocity.u
            y1 = meteo.coord.y + self.dt * meteo.velocity.v
            u1 = meteo.velocity.u + self.dt * meteo.du_dt
            v1 = meteo.velocity.v + self.dt * meteo.dv_dt
            m1 = meteo.mass +self.dt * meteo.dmass_dt

            meteo_for_calc = MeteoriteProperty(t, _Coordinate(x1, y1), _Velocity(u1, v1), m1, meteo.k1, meteo.k2, meteo.tau)
            x = meteo.coord.x + 0.5 * self.dt * (meteo.velocity.u + meteo_for_calc.velocity.u)
            y = meteo.coord.y + 0.5 * self.dt * (meteo.velocity.v + meteo_for_calc.velocity.v)
            u = meteo.velocity.u + 0.5 * self.dt * (meteo.du_dt + meteo_for_calc.du_dt)
            v = meteo.velocity.v + 0.5 * self.dt * (meteo.dv_dt + meteo_for_calc.dv_dt)
            m = meteo.mass + 0.5 * self.dt * (meteo.dmass_dt + meteo_for_calc.dmass_dt)
            
            next_meteo = MeteoriteProperty(t, _Coordinate(x, y), _Velocity(u, v), m, meteo.k1, meteo.k2, meteo.tau)
            self.properties.append(next_meteo)

            if y <= 0:
                break
            if len(self.properties) > 10000000:
                print("Error: too many iterations")
                break
        # print("Calculation finished")

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

        initial_condition = data.iloc[0]
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('Meteorite Properties: y0 = {:.1f} km, u = {:.1f} km/s, v = {:.1f} km/s, m0 = {:.1f} g'.format(
            initial_condition['y']/1e5, initial_condition['u'] / 1e5, initial_condition['v'] / 1e5, initial_condition['mass']
        ))
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

 
    


