from scipy.stats import cauchy, norm, laplace, poisson, uniform
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


sns.set_style('whitegrid')
colors = sns.color_palette('magma', 10)
sizes = [10, 50, 1000]


def plot(dist, name):
    plt.figure(figsize=(15, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        data = dist.rvs(size=sizes[i])
        sns.histplot(data, bins=10, stat="density", color=colors[5])
        if name == 'P(k, 10)':
            x = np.linspace(0, max(data), max(data) + 1)
            y = dist.pmf(x)
        else:
            x = np.linspace(min(data), max(data), 500)
            y = dist.pdf(x)
        n = str(sizes[i])
        if i == 1:
            plt.title(name)
        plt.xlabel(n + ' samples');
        sns.lineplot(x=x, y=y, linewidth=2, color=colors[9])
    plt.show()


plot(norm(loc=0, scale=1), 'N(x,0,1)')
plot(cauchy(), 'C(x,0,1)')
plot(laplace(scale= 1/np.sqrt(2), loc=0), r'L(x,0,$\frac{1}{\sqrt{2}})$')
plot(poisson(10), 'P(k, 10)')
plot(uniform(loc= -np.sqrt(3), scale= 2 * np.sqrt(3)), r'$U(x,-{\sqrt{3}},{\sqrt{3}})$')