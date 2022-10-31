import numpy as np
import math
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
class PolicyGradient:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        np.random.seed(42)
        self.theta = np.random.rand((6))
        self.theta = np.ndarray.tolist(self.theta)
        #self.theta = np.zeros(6)
        self.x = np.random.uniform(0,5)
        self.gamma = 0.90
        self.lr = 0.01
    # REWARD Function
    @staticmethod
    def reward(x):
        if x < 3:
            return 0
        elif x >= 5:
            return +10
        else:
            return 25*math.cos(math.pi*(x-3))-25
    # Noise term w(t) 
    @staticmethod
    def noise():
        return np.random.normal(loc = 0.0, scale = 0.2)
    # Phi function
    @staticmethod
    def phi(x):
        return np.exp(-2 * (x ** 2))
    # Mu : average of action distribution which is the gaussian distribution.
    def mu(self,x,theta):
        phi =  [self.phi(x-z) for z in range(6)]
        mu = np.matmul(phi,theta)
        return mu
    # Followed to above mu, phi -> the stochastic action is selected.
    def action(self,x,theta):
        mu = self.mu(x,theta)
        return np.random.normal(loc = mu, scale = 0.3)
    # To use the REINFORCE algorithm, I should calculate the trajectory.
    def trajectory(self,theta):
        done = True
        A = []
        S = []
        R = []
        x = np.random.uniform(0,5)
        while done:
            if x >= 5:
                done = False
            if x<0: # Sampling again
                x = np.random.uniform(0,5)
                A = []
                S = []
                R = []
            A.append(self.action(x,theta))
            S.append(x)
            R.append(self.reward(x))
            x = x + self.action(x,theta) + self.noise()
        return A,S,R
    # Then we try to learn by using REINFOCE algorithm.
    def REINFORCE(self):
        # Initialize the policy.
        theta = self.theta
        MU = [[],[],[],[],[],[]]
        for _ in range(self.max_iter):
            print(_)
            # generate an episode following policy
            # action, state, and reward. Not return
            theta_old = theta
            A,State,R = self.trajectory(theta_old)
            print(A)
            print(State)
            print(R)
            T = len(A)
            Return = np.zeros(T)
            for i in reversed(range(T-1)):
                x = State[i]
                Return[i] = R[i] + self.gamma * Return[i+1]
                L= [np.exp(-2*((x - k) ** 2))*(x - self.mu(x,theta_old))/(0.3**2) for k in range(6)]
                #L = normalize([LL],axis = 1).flatten()
                for j in range(6):
                    theta[j] = theta_old[j] + self.lr * L[j] *(self.gamma ** i) * Return[i]
                theta = normalize([theta],axis=1).flatten()
            
            print(theta_old)
            MU[0].append(self.mu(0,theta))
            MU[1].append(self.mu(1,theta))
            MU[2].append(self.mu(2,theta))
            MU[3].append(self.mu(3,theta))
            MU[4].append(self.mu(4,theta))
            MU[5].append(self.mu(5,theta))
            
            a = abs(theta - theta_old)
            key = 0
            for i in range(6):
                key += a[i]
            if key/6 < 1e-9:
                break
            print(theta)
        return theta,MU

def main():
    PG = PolicyGradient(200)
    THETA,MU = PG.REINFORCE()
    print(THETA)
    plt.figure(1)
    plt.plot(range(len(MU[0])),MU[0])
    plt.title('mu at position zero')
    plt.figure(2)
    plt.plot(range(len(MU[1])),MU[1])
    plt.title('mu at position one')
    plt.figure(3)
    plt.plot(range(len(MU[2])),MU[2])
    plt.title('mu at position two')
    plt.figure(4)
    plt.plot(range(len(MU[3])),MU[3])
    plt.title('mu at position three')
    plt.figure(5)
    plt.plot(range(len(MU[4])),MU[4])
    plt.title('mu at position four')
    plt.figure(6)
    plt.plot(range(len(MU[5])),MU[5])
    plt.title('mu at position five')
    plt.show()
    # A,S,R have the same length
if __name__ == "__main__":
    main()
    '''
    print(A)
    print(len(A))
    print(S)
    print(len(S))
    print(R)
    print(len(R))
    '''
        # upd_theta =  theta + learning_rate * phi(x-k)
#theta = thetat + learning_rate * reward(s)
# To use the gaussian distribution : I will use the numpy.random.normal

