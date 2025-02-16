def simpsons_method(f: callable, lower_boundary: float, upper_boundary: float, num_of_division: int) -> float:
    """シンプソン法により積分を求める"""
    m = num_of_division // 2
    h = (upper_boundary - lower_boundary) / num_of_division

    integral = 0
    xlow = lower_boundary
    flow = f(xlow)

    for _ in range(1, m):
        # compute other meshpoints of thw successive subintervals and their function values
        xmid = xlow + h
        xup = xlow + 2 * h

        fmid = f(xmid)
        fup = f(xup)

        # compute and add contribution of these two subintervals
        integral += h * (flow  + 4 * fmid + fup) / 3

        # prepare for next iteration
        xlow = xup
        flow = fup
    print(f"Integral of f(x) from {lower_boundary} to {upper_boundary} is {integral}")

    return integral