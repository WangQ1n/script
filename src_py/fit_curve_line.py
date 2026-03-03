import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# 指定一组点
points = np.array([
    [0.0, 9.0],
    [1.0, 8.0],
    [2.0, 9.0],
    [3.0, 9.0],
    [4.0, 4.0],
    [5.0, 9.0],
    [6.0, 8.0],
    [7.0, 9.0],
    [8.0, 8.0],
    [9.0, 7.0],
])

# 贝塞尔曲线生成（使用De Casteljau算法）
def bezier_curve(points, n_times=1000):
    n_points = len(points)
    combinations = np.array([np.math.factorial(n_points - 1) / 
                             (np.math.factorial(i) * np.math.factorial(n_points - 1 - i)) 
                             for i in range(n_points)])
    
    def bezier_interp(t):
        return sum(combinations[i] * (1 - t)**(n_points - 1 - i) * t**i * points[i] for i in range(n_points))
    
    return np.array([bezier_interp(t) for t in np.linspace(0, 1, n_times)])

# B样条曲线生成
def bspline_curve(points, n_times=1000):
    degree = 3  # B样条的阶数
    n_points = len(points)
    knots = np.concatenate(([0] * degree, np.linspace(0, 1, n_points - degree + 1), [1] * degree))
    spl = BSpline(knots, points, degree)
    return spl(np.linspace(0, 1, n_times))

# 拟合的三次多项式曲线生成，轨道更符合三次曲线
def cubic_fit_curve(points, n_times=1000):
    x = points[:, 0]
    y = points[:, 1]
    coefficients = np.polyfit(x, y, 3)
    polynomial = np.poly1d(coefficients)
    x_new = np.linspace(x.min(), x.max(), n_times)
    y_new = polynomial(x_new)
    return np.column_stack((x_new, y_new))

# 生成贝塞尔曲线
bezier_points = bezier_curve(points)

# 生成B样条曲线
bspline_points = bspline_curve(points)

# 生成拟合的三次多项式曲线
cubic_fit_points = cubic_fit_curve(points)

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(points[:, 0], points[:, 1], 'ro--', label='Control Points')
plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'b-', label='Bezier Curve')
plt.plot(bspline_points[:, 0], bspline_points[:, 1], 'g-', label='B-Spline Curve')
plt.plot(cubic_fit_points[:, 0], cubic_fit_points[:, 1], 'm-', label='Cubic Fit Curve')
plt.legend()
plt.title('Bezier, B-Spline, and Cubic Fit Curves')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid()
plt.show()