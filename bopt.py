from matplotlib import pyplot as plt
import numpy as np
coefs = [-1, 0.5, -0.1]

def f(x):
    total = 0
    for exp, coef in enumerate(coefs):
        total += coef * (x ** exp)
    return total

def k(xs, ys, sigma=1, l=1):
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)

def m(x):
    return np.zeros_like(x)

xs = np.linspace(-5.0, 3.5, 100)
ys = f(xs)


x_obs = np.array([-4, -0.5,])
y_obs = f(x_obs)

for iter in range(20):

    x_s = np.linspace(-8, 7, 80)

    K = k(x_obs, x_obs)
    K_s = k(x_obs, x_s)
    K_ss = k(x_s, x_s)

    K_sTKinv = np.matmul(K_s.T, np.linalg.pinv(K))

    mu_s = m(x_s) + np.matmul(K_sTKinv, y_obs - m(x_obs))
    Sigma_s = K_ss - np.matmul(K_sTKinv, K_s)

    y_true = f(x_s)

    plt.figure(0)
    plt.subplot(2,1,1)
    plt.xlim(-8, 8)
    #plt.ylim(-7, 8)
    plt.plot(xs, ys, color='green')
    plt.plot(x_s, y_true, color='blue')
    plt.plot(x_s, mu_s, color='black')
    stds = np.sqrt(Sigma_s.diagonal() + 1e-8)
    err_xs = np.concatenate((x_s, np.flip(x_s, 0)))
    err_ys = np.concatenate((mu_s + 2 * stds, np.flip(mu_s - 2 * stds, 0)))
    plt.fill_between(err_xs, err_ys)
    for i in range(5):
        y_s = np.random.multivariate_normal(mu_s, Sigma_s)
        plt.plot(x_s, y_s)
    plt.plot(x_obs[-1], y_obs[-1], marker='o', mec='r', mfc='w', ms=10)

    ucb_v = mu_s + 2 * stds
    max_idx = np.argmax(ucb_v)
    print ucb_v
    print max_idx
    plt.subplot(2,1,2)
    plt.plot(x_s, ucb_v)
    plt.plot(x_s[max_idx], ucb_v[max_idx], marker='o', mec='r', mfc='black', ms=10)
    plt.show()
    x_a = np.array([x_s[max_idx]])
    print x_a
    x_obs = np.concatenate((x_obs, x_a))
    y_obs = f(x_obs)
    print x_obs
