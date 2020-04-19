import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
np.seterr(divide='ignore', invalid='ignore') # ignore a warning for dividing to zero resulting INF
class SecondPaper:
    def __init__(self, a=100, b=1, c=20, n_min=5, n_max=100, lamb_max=1, lamb_min=0.05, number_of_lamb=20):
        self.a = a
        self.b = b
        self.c = c
        self.n_firms = np.array(range(n_min,n_max+1,1)).reshape(-1,1) # total number of firms
        self.lamb = np.asarray(np.linspace(lamb_min, lamb_max, number_of_lamb))[:, np.newaxis] # Enforceability level
        self.nn,_ = np.meshgrid(self.n_firms, self.lamb)
        _,self.ll = np.meshgrid(self.n_firms, self.lamb)
        self.first_coef = self.a - self.c
        self.second_coef = self.first_coef/self.b
        self.third_coef = self.first_coef**2/self.b
        self.d1 = self.n_C().shape[0] # first dimension of array n_C (number of rows)
        self.d2 = self.n_C().shape[1] # second dimension of array n_C (number of columns)
        self.n_max = n_max
        self.n_min = n_min
        self.lamb_min = lamb_min
        self.lamb_max = lamb_max

    def n_C(self):
        return np.floor((self.nn + 2*self.ll + 1)/(self.ll + 1))

    def market_volume(self):
        n_C = self.final_stable_nC()
        market_volume = self.second_coef*(1 - 1/(self.nn - n_C + 1) + n_C/(np.multiply((self.nn-n_C+1), (n_C + 1 + np.multiply(self.ll,(n_C-1))))))
        return market_volume

    def market_price(self):
        n_C = self.final_stable_nC()
        P = self.first_coef*(1 + np.multiply(self.ll,(n_C-1)))/(np.multiply((self.nn-n_C+1), (n_C + 1 + np.multiply(self.ll,(n_C-1))))) + self.c
        return P

    def member_profit(self, n_C):
        return self.third_coef*(1 + np.multiply(self.ll, (n_C - 1)))/(np.multiply((self.nn-n_C+1), (n_C + 1 + np.multiply(self.ll,(n_C-1)))**2))

    def fringe_profit(self, n_C):
        return self.third_coef*(1 + np.multiply(self.ll, (n_C - 1)))**2/(np.multiply((self.nn-n_C+1)**2, (n_C + 1 + np.multiply(self.ll,(n_C-1)))**2))

    def internal_stability(self, n_C):
        return self.member_profit(n_C) - self.fringe_profit(n_C-1)

    def external_stability(self, n_C):
        return self.member_profit(n_C +1) - self.fringe_profit(n_C)

    def stable_nC1(self):
        n_C = self.n_C()
        in_stability = self.internal_stability(n_C)
        ex_stability = self.external_stability(n_C)
        for i in range(self.d1):
            for j in range(self.d2):
                if in_stability[i,j] < 0 or ex_stability[i,j] > 0: # check stability conditions
                    n_C[i,j] = None
        return n_C

    def stable_nC2(self):
        n_C2 = self.n_C() + 1
        for i in range(self.d1):
            for j in range(self.d2):
                if self.nn[i, j] <= n_C2[i, j]: # check whether n_C > n
                    n_C2[i, j] = wp2.nn[i, j]
        in_stability2 = self.internal_stability(n_C2)
        ex_stability2 = self.external_stability(n_C2)
        for i in range(self.d1):
            for j in range(self.d2):
                if in_stability2[i, j] < 0 or ex_stability2[i, j] > 0: # check stability conditions
                    n_C2[i, j] = None
        return n_C2

    def stable_nC3(self):
        n_C3 = self.n_C() + 2
        for i in range(self.d1):
            for j in range(self.d2):
                if self.nn[i, j] <= n_C3[i, j]: # check whether nC > n
                    n_C3[i, j] = self.nn[i, j]
        in_stability3 = self.internal_stability(n_C3)
        ex_stability3 = self.external_stability(n_C3)
        for i in range(self.d1):
            for j in range(self.d2):
                if in_stability3[i, j] < 0 or ex_stability3[i, j] > 0: # check stability conditions
                    n_C3[i, j] = None
        return n_C3

    def final_stable_nC(self):
        stable_nC = self.stable_nC1()
        n_C2 = self.stable_nC2()
        n_C3 = self.stable_nC3()
        for i in range(self.d1):
            for j in range(self.d2):
                if np.isnan(stable_nC[i, j]) and not np.isnan(n_C2[i, j]):
                    stable_nC[i, j] = n_C2[i, j]
                elif np.isnan(stable_nC[i, j]) and not np.isnan(n_C3[i, j]):
                    stable_nC[i, j] = n_C3[i, j]
                elif np.isnan(stable_nC[i, j]) or stable_nC[i, j] > j + self.n_min:
                    stable_nC[i, j] = j + self.n_min # fill the data with a complete cartel for the other cases
        return stable_nC

    def resulting_nF(self):
        return self.nn - self.final_stable_nC()

    def final_data(self):
        data = pd.DataFrame(np.concatenate((self.lamb, self.final_stable_nC()), axis=1))
        data.columns = ["lambda"] + ["n=" + str(k) for k in range(self.n_min, self.n_max+1, 1)]
        return data

    def legend_array(self):
        legend_array = np.array([35, 25, 15])*(int(self.n_max/50))
        return legend_array

    def middle_value(self):
        return (2*self.nn + self.ll + 3.5)/(2*self.ll + 2)

    def surface_data(self):
        surface_data = self.final_stable_nC()
        middle_value = self.middle_value()
        for i in range(self.d1):
            for j in range(self.d2):
                if surface_data[i, j] < j + self.n_min:
                    surface_data[i, j] = middle_value[i, j]
        return surface_data

      def index_list(self, n_C):
        start = 0
        i = len(n_C) - 1
        index_list = []
        while i >= 0:
            if start != n_C[i]:
                start = n_C[i]
                index_list.append(i)
                continue
            i -= 1
        return index_list

    def welfare_plot(self, n):
        n_C = self.final_stable_nC()[:, n - self.n_min]
        index_list = self.index_list(n_C)
        lambd = self.lamb.flatten()
        Q = self.market_volume()[:, n - self.n_min]
        fig = plt.figure(figsize=(19.20, 12.80))
        ax = fig.add_subplot(111)
        plt.rcParams['font.family'] = 'Times New Roman'
        ax.set_xlabel('Enforceability Level $\lambda$', fontsize=20)
        ax.set_ylabel('Total Output $Q$', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        ax.set_xlim(-0.05, 1.05)
        plt.style.use('seaborn-whitegrid')
        plt.plot(lambd, Q, "o-k")
        for i in index_list:
            ax.text(lambd[i] + 0.01, Q[i], s='$k$ = {}'.format(int(n_C[i])), fontsize=20)
        plt.savefig('Welfare.png', bbox_inches='tight', pad_inches=0)
        plt.show()


    def welfare_animation(self, n):
        n_C = self.final_stable_nC()[:, n - self.n_min]
        index_list = self.index_list(n_C)
        lambd = self.lamb.flatten()
        Q = self.market_volume()[:, n - self.n_min]
        fig = plt.figure(figsize=(19.20, 12.80))
        ax = fig.add_subplot(111)
        plt.style.use('seaborn-whitegrid')
        def update(ii):
            i = len(lambd) - ii -1
            j = len(lambd)
            # title = 'Q = %.2f' % (Q[i]) + ', $\lambda = %.2f$' % (lambd[i]) + ', $n$ = 10'
            xlabel = 'Enforceability Level $\lambda$'
            ylabel = "Total Quantity $Q$"

            animlist = plt.cla()
            animlist = plt.axis([-0.02, np.max(lambd)+0.02, np.min(Q) - 1, np.max(Q) + 1])
            # animlist = plt.plot([], [])
            animlist = plt.plot(lambd[i:j], Q[i:j], 'ko-')
            ax.set_xlabel(xlabel, fontsize=20)
            ax.set_ylabel(ylabel, fontsize=20)
            # ax.set_title(title, fontsize = 20)
            ax.set_title('Q = %.2f' % (Q[i]) + ', $n = 10$', fontsize=30)
            ax.tick_params(axis='both', which='major', labelsize=18)
            ax.tick_params(axis='both', which='minor', labelsize=14)
            for k in reversed(index_list):
                if k >=i:
                    text = '$k$ = %d' % (n_C[k])
                    ax.text(lambd[k] + 0.01, Q[k], s='$k$ = {}'.format(int(n_C[k])), fontsize=20)
                    ax.text(lambd[k] + 0.01, Q[k] - 0.5, s='$\lambda$ = {:.2f}'.format(lambd[k]), fontsize=20)
                    ax.text(lambd[k] + 0.01, Q[k] - 1, s='$Q$ = {:.2f}'.format(Q[k]), fontsize=20)
            # ax.text(lambd[i], np.max(Q) + 0.5, s='Q = %.2f' % (Q[i]), fontsize=30)
            # ax.text(np.max(lambd) - 0.3, (np.max(Q) + np.min(Q)) / 2, s='Q = %.2f' % (Q[i]), fontsize=40)

            return animlist, ax

        anim = FuncAnimation(fig, update, frames=np.arange(0, 21), interval=1000)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
        anim.save('wp2.mp4', writer=writer)
        anim.save('wp2.gif', dpi=100, writer='imagemagick')
        plt.show()

    def plot_2D(self): # best for case of n=50
        plt.figure()
        plt.scatter(self.ll, self.nn, s=self.n_C()/(int(self.n_max/50)), cmap='viridis', alpha=0.5, c='k')
        plt.title("Number of Cartel Members")
        plt.xlabel("Enforceability level")
        plt.ylabel("Total number of firms")
        for member in self.legend_array():
            plt.scatter([], [], c='k', alpha=0.5, s=member, label=str(member) + " members") # add another empty scatter
            # plot to create a desired legend
        plt.legend(loc="center left", scatterpoints=1, labelspacing=1, title="$n_C$", frameon=False,
                   bbox_to_anchor=(1, 0.5)) # bbox_to_anchor to adjust the position of legend outside
        # (use for unusual position)
        plt.show()

wp2 = SecondPaper(n_max=50, n_min=2, lamb_max=1, lamb_min=0, number_of_lamb=50)

wp2.bar3D_plot()
wp2.suface_plot()
wp2.welfare_plot(10)
