import math
from typing import Literal
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class _Coordinate:
    x: float
    y: float
    
class EquipotentialCurves:
    """2連星系の等ポテンシャル面を扱うクラス"""
    mu: float # 2連星系の質量比
    dx: float # 等ポテンシャル面を求めるときのxの変化量: float
    dy: float # 等ポテンシャル面を求めるときのyの変化量: float

    def __init__(self, mu: float, dx: float = 0.001, dy: float = 0.001):
        self.mu = mu
        self.dx = dx
        self.dy = dy

    def r1(self, coord: _Coordinate) -> float:
        """primary1からの距離"""
        return math.sqrt((coord.x - self.mu)**2 + coord.y**2)
    
    def r2(self, coord: _Coordinate) -> float:
        """primary2からの距離"""
        return math.sqrt((coord.x + 1 - self.mu)**2 + coord.y**2)

    def vxy(self, coord: _Coordinate) -> list[float]:
        """coordにおけるポテンシャル"""
        if coord.y == 0:
            return [
                h1 * (1 - self.mu) / (coord.x - self.mu) + h2 * self.mu / (coord.x + 1 - self.mu) + coord.x**2 / 2 
                for h1, h2 in [(-1, 1), (-1, -1), (1, 1)]
            ]
        else:
            return [(1 - self.mu) / self.r1(coord) + self.mu / self.r2(coord) + 1/2 * (coord.x**2 + coord.y**2)]
    
    def dvdx(self, coord: _Coordinate) -> list[float]:
        """coordにおけるポテンシャルのxに関する微分dv/dx"""
        if coord.y == 0:
            return [
                -h1 * (1 - self.mu) / (coord.x - self.mu)**2 - h2 * self.mu / (coord.x + 1 - self.mu)**2 + coord.x
                for h1, h2 in [(-1, 1), (-1, -1), (1, 1)]
            ]
        else:
            return [-(1 - self.mu) / self.r1(coord)**2 - self.mu / self.r2(coord)**2 + coord.x]
    
    def dvdy(self, coord: _Coordinate) -> float:
        """coordにおけるポテンシャルのyに関する微分dv/dy"""
        return -(1 - self.mu) * coord.y / self.r1(coord)**3 - self.mu * coord.y / self.r2(coord)**3 + coord.y
    
    def d2vdx2(self, coord: _Coordinate) -> list[float]:
        """coordにおけるポテンシャルのxに関する2階微分d^2v/dx^2"""
        if coord.y == 0:
            return [
                2*h1*(1 - self.mu)/(coord.x - self.mu)**3+ 2*h2*self.mu/(coord.x + 1 - self.mu)**3 + 1
                for h1, h2 in [(-1, 1), (-1, -1), (1, 1)]
            ]
        else:
            return [2*(1 - self.mu)/self.r1(coord)**3 + 2*self.mu/self.r2(coord)**3 + 1]
    
    @property
    def L1(self) -> _Coordinate:
        """ラグランジュ点L1"""
        x = 0
        for _ in range(10000):
            dvdx = self.dvdx(coord=_Coordinate(x, 0))
            d2vdx2 = self.d2vdx2(coord=_Coordinate(x, 0))
            assert len(dvdx) == 3 and len(d2vdx2) == 3
            x = x - dvdx[0]/ d2vdx2[0]
        return _Coordinate(x=x, y=0)
    
    @property
    def L2(self) -> _Coordinate:
        """ラグランジュ点L2"""
        x = -1
        for _ in range(10000):
            dvdx = self.dvdx(coord=_Coordinate(x, 0))
            d2vdx2 = self.d2vdx2(coord=_Coordinate(x, 0))
            assert len(dvdx) == 3 and len(d2vdx2) == 3
            x = x - dvdx[1] / d2vdx2[1]
        return _Coordinate(x=x, y=0)
    
    @property
    def L3(self) -> _Coordinate:
        """ラグランジュ点L3"""
        x = 1
        for _ in range(10000):
            dvdx = self.dvdx(coord=_Coordinate(x, 0))
            d2vdx2 = self.d2vdx2(coord=_Coordinate(x, 0))
            assert len(dvdx) == 3 and len(d2vdx2) == 3
            x = x - dvdx[2] / d2vdx2[2]
        return _Coordinate(x=x, y=0)
    
    def equipotential_curve(self, coord: _Coordinate) -> list[_Coordinate]:
        curve = [coord]
        now_potential = self.vxy(coord)[0]
        is_dx_update = True
        for _ in tqdm(range(1000)):
            x = curve[-1].x
            y = curve[-1].y
            if is_dx_update:
                x12 = x + 0.5 * self.dx
                y12 = y - 0.5 * self.dx * self.dvdx(_Coordinate(x,y))[0] / self.dvdy(_Coordinate(x,y))
                x = x + self.dx
                y = y - self.dx * self.dvdx(_Coordinate(x=x12, y=y12))[0] / self.dvdy(_Coordinate(x=x12, y=y12))
                y = y - (self.vxy(_Coordinate(x=x, y=y))[0] - now_potential) / self.dvdy(_Coordinate(x=x, y=y))
            else:
                y12 = y + 0.5 * self.dy
                x12 = x - 0.5 * self.dy * self.dvdy(_Coordinate(x,y)) / self.dvdx(_Coordinate(x,y))[0]
                y = y + self.dy
                x = x - self.dy * self.dvdy(_Coordinate(x=x12, y=y12)) / self.dvdx(_Coordinate(x=x12, y=y12))[0]
                x = x - (self.vxy(_Coordinate(x=x, y=y))[0] - now_potential) / self.dvdx(_Coordinate(x=x, y=y))[0]
            coord = _Coordinate(x=x, y=y)
            potential = self.vxy(coord)[0]
            if abs(now_potential - potential) >= 1e-2:
                break
            curve.append(coord)
            if self._is_close(coord, curve[0]):
                break

            # 前回からの変化量を計算
            dx_ = curve[-1].x - curve[-2].x
            dy_ = curve[-1].y - curve[-2].y
            if abs(dx_) > abs(dy_):
                is_dx_update = True
                self.dx = abs(self.dx) if dx_ > 0 else -abs(self.dx)
            else:
                is_dx_update = False
                self.dy = abs(self.dy) if dy_ > 0 else -abs(self.dy)
        return curve
    
    def _is_close(self, p1: _Coordinate, p2: _Coordinate) -> bool:
        return abs(p1.x - p2.x) < 1e-5 and abs(p1.y - p2.y) < 1e-5