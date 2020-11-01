"""
Student number: 0589870
Student name: Dmitrii Shumilin
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA


class FaultDetector:
    """
    Needed to be imported Numpy and Pandas
    Class to perform fault detection over dataset of time-series data with few variables
    
    Initialization with:
    data_norm - 'matrix', dataset with normal behavior of process
    data_fault - 'matrix', dataset for fault detection
    type_PCA - 'string', type of analysis. Can be neither PCA or DPCA, default = PCA
    L - 'int' number of lags for DPCA, usually calculation can be done automatically, default = None
    rolling - 'bool', enable rolling window over dataset, default = False
    window - 'int', size of rolling window, default = 10
    random_seed - 'int', seed to reproduce result, default = 2022
    
    Methods:
    self.hotelling_statistic - return T2 row of statistics
    self.hotelling_treshold(alpha) - return treshold for T2 statistic with alpha (level of significance)
    self.q_statistic - return Q2 row statistics
    self.q_treshold(alpha) - return treshold for Q2 statistic with alpha (level of significance)
    self.augmentation(L, data) - return dataset with L lags in data
    
    
    Calculations based on:
    [1] Evan L. Russell, Leo H. Chiang, Richard D. Braatz, Fault detection in industrial processes using canonical
    variate analysis and dynamic principal component analysis, 2000, https://doi.org/10.1016/S0169-7439(00)00058-7.
    [2] Wenfu Ku, Robert H. Storer, Christos Georgakis, Disturbance detection and isolation by dynamic principal
    component analysis, 1995, https://doi.org/10.1016/0169-7439(95)00076-3.
    """

    def __init__(self, data_norm, data_fault, type_PCA='PCA', L=None, rolling=False, window=10, random_seed=2022):
        # Defence from other methods
        self.mode = ['PCA', 'DPCA']
        assert type_PCA in self.mode, 'Mode must be neither PCA or DPCA'
        self.type_pca = type_PCA
        
        # Store data inside of class
        self.norm = data_norm.copy()
        self.fault = data_fault.copy()
        
        # If Rolling is True, then use rolling average window 
        if rolling:
            self.norm = self.norm.rolling(window=window, center=False).mean().dropna(axis=0)
            self.fault = self.fault.rolling(window=window, center=False).mean().dropna(axis=0)

        # Store mean and std of normal data variables
        self.mean = self.norm.mean(axis=0)
        self.std = self.norm.std(axis=0)
        
        # Fix seed
        np.random.seed(random_seed)
        
        # Data can come in numpy array or pandas DataFrame
        # Normalize normal dataset
        if isinstance(self.norm, pd.DataFrame):
            self.data0 = ((self.norm - self.mean) / self.std).to_numpy()
        else:
            self.data0 = ((self.norm - self.mean) / self.std)
        # Normalize fault dataset
        if isinstance(self.fault, pd.DataFrame):
            self.data1 = ((self.fault - self.mean) / self.std).to_numpy()
        else:
            self.data1 = ((self.fault - self.mean) / self.std)
        
        # Make augmentation over datasets if type_pca is 'DPCA'
        if self.type_pca == 'DPCA':
            # If L needed to be specified manually
            if L:
                self.L = L
                self.data1 = self.augmentation(self.L, self.data1.copy())
            else:
                self.L, self.data1 = self._lags_num()  # Receive number of lags and augmented fault dataset
            self.data0 = self.augmentation(self.L, self.data0.copy())  # Receive augmented normal dataset
            
        self.n = self.data1.shape[0]  # Number of observations
        self.a = self._get_a_number()  # Receive new number of dimensions for PCA decomposition

    def _get_a_number(self):
        """
        Calculate new number of dimensions for PCA decomposition based on parallel analysis.
        """
        # Almost independent observations is simulated by random gaussian noise
        noise = np.random.normal(size=self.data1.shape)
        _, s_noise, _ = np.linalg.svd(noise)  # SVD decomposition
        _, s_fault, _ = np.linalg.svd(self.data1)  # SVD decomposition
        a = np.argmin(abs(s_fault - s_noise)) + 1  # Find crossing point on the plot of singular values
        return a

    def hotelling_statistic(self):
        """
        Calculation of hotelling (T2) statistic, based on the work [1]
        """
        pca = PCA(self.a)  # Initialization of PCA decomposition
        pca.fit(self.data1/np.sqrt(self.n-1))  # Find principal components
        s = np.diag(pca.singular_values_)  # Get truncated matrix of singular values
        sig2 = np.linalg.inv(s).dot(np.linalg.inv(s))
        P = pca.components_.T  # Get principal components

        time_series_statistic = []  # Basket
        for i in range(self.n):
            x = self.data1[i, :]
            stat = x.T.dot(P).dot(sig2).dot(P.T).dot(x)
            time_series_statistic.append(stat)  # Add data into the basket
            
        return time_series_statistic

    def hotelling_treshold(self, alpha):
        """
        Calculation threshold of T2 staticstic based on the Fisher distribution
        """
        mult = (((self.n**2) - 1) * self.a) / (self.n * (self.n - self.a))
        crit_val = stats.f.ppf(q=1-alpha/2, dfn=self.a, dfd=self.n-self.a)  # Critical value for F distribution

        return mult * crit_val

    def q_statistic(self):
        """
        Calculation Q2 statistic (Residuals) for fault dataset based on theory in [1]
        """
        pca = PCA(self.a)  # Initialization of PCA decomposition
        pca.fit(self.data0)  # Find principal components
        P0 = pca.components_.T  # Get principal components

        T4 = self.data1.dot(P0)  # Project data to normal principal components

        Q = self.data1 - T4.dot(P0.T)
        Q = Q.T                     # Calculation statistic
        Q = np.sum(Q**2, axis=0)
        return Q
    
    def q_treshold(self, alpha):
        """
        Calculation threshold of Q2 statistic
        """
        _, s, _ = np.linalg.svd(self.data1/np.sqrt(self.n-1))

        th1 = np.sum(s[self.a+1:]**2)
        th2 = np.sum(s[self.a+1:]**4)
        th3 = np.sum(s[self.a+1:]**6)
        h0 = 1 - (2*th1*th3)/(3*th2**2)

        ca = stats.norm.ppf(1-alpha)

        q = th1 * ((h0*ca*np.sqrt(2*th2))/th1 + 1 + (th2*h0*(h0-1))/th1**2)**(1/h0)
        return q

    def _lags_num(self):
        """
        Calculation of appropriate number of lags based on algorithm proposed in [2]
        L - number of lags
        r - rank of matrix
        """
        
        r_new = np.zeros(10)
        # Step 1
        L = 1
        while True:

            # Step 2
            augmented = self.augmentation(L, self.data1.copy())

            # Step 3
            _, s_fault, _ = np.linalg.svd(augmented)

            # Step 4
            j = self.data1.shape[1]*L-1
            r = 0

            # Step 5 and 6, choose threshold 0.01
            while s_fault[j] < 0.01:
                j = j - 1
                r = r + 1

            # Step 7
            summa = 0
            for i in range(L):
                summa += (L-i+1)*r_new[i] 
            r_new[L] = r - summa

            # Step 8
            if r_new[L] <= 0:
                break

            # Step 9
            L = L + 1
        return L, augmented

    def augmentation(self, L, data):
        """
        Create a lagged dataframe with number of lag L over Data
        :param L: 'int', Number of lags
        :param data: 'np.array', dataset
        :return:
       """
        augmented = []  # Basket
        for t in range(data.shape[0]-L+1):
            aux = data[t:t+L, :].ravel()  # One row in new dataset
            augmented.append(aux)  # Add row into the basket
        augmented = np.array(augmented)  # Convert also to np.array
        return augmented
