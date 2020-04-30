import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

np.seterr(divide='ignore', invalid='ignore')  # ignore a warning for dividing to zero resulting INF


class SecondPaper:
    def __init__(self, a=100, b=1, c=20, n_min=5, n_max=100, lamb_max=1, lamb_min=0.05, number_of_lamb=20, number_of_n=10):
        self.a = a
        self.b = b
        self.c = c
        self.n_max = n_max
        self.n_min = n_min
        self.lamb_min = lamb_min
        self.lamb_max = lamb_max
        self.number_of_lamb = number_of_lamb
        self.number_of_n = self.n_max - self.n_min + 1
        if self.number_of_n < number_of_n:
            self.n_firms = np.array(np.linspace(n_min, n_max, number_of_n)).reshape(-1, 1)
        else:
            self.n_firms = np.array(np.linspace(self.n_min, self.n_max, self.number_of_n)).reshape(-1,
                                                                                                   1)  # total number of firms
        self.lamb = np.asarray(np.linspace(self.lamb_min, self.lamb_max, self.number_of_lamb))[:,
                    np.newaxis]  # Enforceability level
        self.first_limit = (2 + np.sqrt((self.lamb + 1) ** 2  + 4 * (self.lamb + 1))) /self.lamb - 1
        self.nn, _ = np.meshgrid(self.n_firms, self.lamb)
        _, self.ll = np.meshgrid(self.n_firms, self.lamb)
        self.first_coef = self.a - self.c
        self.second_coef = self.first_coef / self.b
        self.third_coef = self.first_coef ** 2 / self.b
        self.d1 = self.k_int().shape[0]  # first dimension of array n_C (number of rows)
        self.d2 = self.k_int().shape[1]  # second dimension of array n_C (number of columns)

    def k_ext(self):
        return (2 * (self.nn + 1) + self.ll * (self.nn - 1) - np.sqrt(
            (self.nn + 1) ** 2 * self.ll ** 2 - 4 * (self.nn + 2) * self.ll)) / (2 * (self.ll + 1))

    def k_ext_2(self):
        return (2 * (self.nn + 1) + self.ll * (self.nn - 1) + np.sqrt(
            (self.nn + 1) ** 2 * self.ll ** 2 - 4 * (self.nn + 2) * self.ll)) / (2 * (self.ll + 1))

    def k_int_limit(self):
        return (2 * (self.first_limit + 2) + self.lamb * (self.first_limit + 1) - np.sqrt(
            (self.first_limit + 1) ** 2 * self.lamb ** 2 - 4 * (self.first_limit + 2) * self.lamb)) / (2 * (self.lamb + 1))

    def k_int(self):
        return (2 * (self.nn + 2) + self.ll * (self.nn + 1) - np.sqrt(
            (self.nn + 1) ** 2 * self.ll ** 2 - 4 * (self.nn + 2) * self.ll)) / (2 * (self.ll + 1))

    def k_int1(self):
        return (2 * (self.final_n_values() + 2) + self.ll * (self.final_n_values() + 1) - np.sqrt(
            (self.final_n_values() + 1) ** 2 * self.ll ** 2 - 4 * (self.final_n_values() + 2) * self.ll)) / (2 * (self.ll + 1))

    def k_stern(self):
        return np.floor(self.k_int())

    def final_k_stern(self):
        final_k_stern = np.copy(self.k_stern())
        for i in range(self.nn.shape[1]):
            final_k_stern[:, i][np.isnan(final_k_stern[:, i])] = self.n_min + i
            for j in range(self.nn.shape[0]):
                if final_k_stern[j, i] > self.n_min + i:
                    final_k_stern[j, i] = self.n_min + i
        return final_k_stern

    def data_line_3D(self):
        return (2 * (self.first_limit + 2) + self.lamb * (self.first_limit + 1) - np.sqrt(
            (self.first_limit + 1) ** 2 * self.lamb ** 2 - 4 * (self.first_limit + 2) * self.lamb)) / (
                           2 * (self.lamb + 1))

    def final_k_stern_values(self):
        final_values = np.copy(self.final_k_stern())
        lambd = np.copy(self.ll)
        final_values[final_values<1/lambd] = 0
        return final_values

    def final_lambd_values(self):
        final_values =  np.copy(self.ll)
        final_k_stern = np.copy(self.final_k_stern_values())
        final_values[np.isnan(final_k_stern)] = np.nan
        return final_values

    def final_n_values(self):
        final_values =  np.copy(self.nn)
        final_k_stern = np.copy(self.final_k_stern_values())
        final_values[np.isnan(final_k_stern)] = np.nan
        return final_values

    def cartel_profit(self, f):
        return (self.ll*(self.a - self.c)**2)/(self.b*(self.ll + 1)**2*(self.nn - self.final_k_stern_values() + 1)*self.final_k_stern_values()) - f

    def k_z(self):
        return self.nn - 2*(np.sqrt((self.nn + 1) ** 2 * self.ll ** 2 - 4 * (self.nn + 2) * self.ll)) / (2 * (self.ll + 1))

    def fix_cost_data(self, f):
        fix_cost_data = np.copy(self.cartel_profit(f))
        fix_cost_data[np.isinf(fix_cost_data)] = -1
        return fix_cost_data


    def bar3D_plot(self, f):
        def change_color(x):
            for i in range(len(x)):
                if x[i] == 1 or x[i] == 0:
                    x[i] = 0.4
                else:
                    x[i] = 1
            return x
        fig = plt.figure(figsize=(19.20, 12.80))
        ax = fig.add_subplot(111, projection='3d')
        ax.force_zorder = True
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        plt.rcParams['font.family'] = 'Times New Roman'
        colors= plt.cm.gray(change_color(self.final_k_stern_values().flatten()/self.nn.flatten()))
        fix_cost_data = self.fix_cost_data(f)
        fix_cost_plot_data = np.copy(self.final_k_stern_values())
        fix_cost_plot_data[fix_cost_data < 0] = 0
        # ls = LightSource(azdeg=0, altdeg=65)
        # rgb = ls.shade(self.final_k_stern_values(), plt.cm.RdYlBu)
        ax.bar3d(self.ll.flatten(), self.nn.flatten(), np.zeros(self.final_k_stern().size),
                 0.02*np.ones(self.final_k_stern().size), np.ones(self.final_k_stern().size),
                 fix_cost_plot_data.flatten(), shade=True, color=colors, linewidth=0, edgecolor='w', facecolors="w")
        ax.set_xlabel('$\lambda$', fontsize=20)
        ax.set_ylabel('$n$', fontsize=20)
        ax.set_zlabel('$k$', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_ylim3d(0, self.n_max)
        ax.set_zlim3d(1, self.n_max)
        # ax.set_xlim3d(self.lamb_min, self.lamb_max + self.lamb_min)
        ax.view_init(15, -50)
        # plt.savefig('bar_3D1.png', bbox_inches='tight', pad_inches=0)
        plt.show()

    def suface_plot(self):
        fig = plt.figure(figsize=(19.20, 12.80))
        ax = fig.gca(projection='3d')
        plt.rcParams['font.family'] = 'Times New Roman'
        ax.set_xlabel('$\lambda$', fontsize=20)
        ax.set_ylabel('$n$', fontsize=20)
        ax.set_zlabel('$k$', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.plot_surface(self.ll, self.nn, self.final_k_stern_values())
        # ax.plot_wireframe(self.lamb, self.first_limit, self.data_line_3D(), color='k', linewidth=2.5)
        ax.set_ylim3d(0, self.n_max)
        ax.set_zlim3d(1, self.n_max)
        ax.view_init(50, -25)
        # plt.savefig('Surface_3D.png', bbox_inches='tight', pad_inches=0)
        plt.show()


wp2 = SecondPaper(n_max=50, n_min=2, lamb_max=1, lamb_min=0.02, number_of_lamb=50, number_of_n=10)
a = wp2.fix_cost_data(5)
b = wp2.final_k_stern_values()
b[a < 0] = np.nan
# a[np.isinf(a)] = -1

# wp2.suface_plot()
wp2.bar3D_plot(5)
