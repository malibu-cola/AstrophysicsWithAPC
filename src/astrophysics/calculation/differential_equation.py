class DifferentialEquation:
    """数値解法により微分方程式の解を求めるクラス"""

    def __init__(self, x0: float, y0: float, dx: float, iter_num: int, f: callable):
        self.x0 = x0
        self.y0 = y0
        self.dx = dx
        self.iter_num = iter_num
        self.f = f

    def eulers_method(self) -> tuple[(float, float)]:
        """オイラー法により微分方程式の解を求める"""
        x = self.x0
        y = self.y0
        for i in range(self.iter_num):
            x1 = x + self.dx
            y1 = y + self.dx * self.f(x, y)

            print(f"{i}\t{x1}\t{y1}")

            x = x1
            y = y1
        return (x, y)

    def cauchy_method(self) -> tuple[(float, float)]:
        """コーシー法により微分方程式の解を求める"""
        x = self.x0
        y = self.y0
        for i in range(self.iter_num):
            x12 = x + 0.5 * self.dx
            y12 = y + 0.5 * self.dx * self.f(x, y)

            x1 = x + self.dx
            y1 = y + self.dx * self.f(x12, y12)

            print(f"{i}\t{x1}\t{y1}")

            x = x1
            y = y1
        return (x, y)

    def heun_method(self) -> tuple[(float, float)]:
        """ホイン法(予測子修正子法)により微分方程式の解を求める"""
        x = self.x0
        y = self.y0
        for i in range(self.iter_num):
            x1p = x + self.dx
            y1p = y + self.dx * self.f(x, y)

            x1 = x + self.dx
            y1 = y + 0.5 * self.dx * (self.f(x, y) + self.f(x1p, y1p))

            print(f"{i}\t{x1}\t{y1}")

            x = x1
            y = y1
        return (x, y)

    def runge_kutta_method(self) -> tuple[(float, float)]:
        """ルンゲ・クッタ法により微分方程式の解を求める"""
        x = self.x0
        y = self.y0
        for i in range(self.iter_num):
            k1 = self.dx * self.f(x, y)
            k2 = self.dx * self.f(x + 0.5 * self.dx, y + 0.5 * k1)
            k3 = self.dx * self.f(x + 0.5 * self.dx, y + 0.5 * k2)
            k4 = self.dx * self.f(x + self.dx, y + k3)

            x1 = x + self.dx
            y1 = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

            print(f"{i}\t{x1}\t{y1}")

            x = x1
            y = y1
        return (x, y)


def newton_raphson_method(
    f: callable, dfdx: callable, x0: float, iter_num: int
) -> float:
    """ニュートン・ラフソン法により非線形方程式の解を求める"""
    x = x0
    for i in range(iter_num):
        x1 = x - f(x) / dfdx(x)
        print(f"{i}\t{x1}")
        x = x1
    return x
