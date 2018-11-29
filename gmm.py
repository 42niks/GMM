import sys
import numpy as np
from scipy.stats import multivariate_normal as mn
from sklearn.preprocessing import normalize
import math
from pathlib import Path
from kmeans import kmeans
from timeit import default_timer as timer
import scipy.io
import warnings
warnings.filterwarnings('error')
# default module wide tolerance for covariance
def_sigma_tol = 1e-3
'''
################################################################################
Variables:
1. X: ndarray, (N, D)
	Stores the input data points. N D-dimensional points
2. mean: ndarray, (K, D)
	Stores the means of all the K clusters
3. sigma: ndarray, (K, D)
	Stores the diagonal covariance matrix of the K clusters
4. pi: ndarray, (K,)
	Stores the mixture-coefficients of the clusters
	Property: normalized as sum(pi) = 1 
5. resp: ndarray, (N, K)
	Stores the responsibilities of K clusters for each of the N points
	Property: normalized as sum_over_clusters(resp[n]) = 1
################################################################################

NOTE: Try to follow this order while passing parameters:
1. The parameter that needs to be modified
2. Rest of the parameters in the following order:
	X > mean > sigma > pi > resp

'''

def _cal_resp(resp, X, mean, sigma, pi):
	K = mean.shape[0]
	N = X.shape[0]
	# print('X shape', X.shape)
	# print('mean shape',mean.shape )
	# print('sigma shape', sigma.shape)
	# print('pi shape', pi.shape)
	# print('resp shape',resp.shape)
	for k in range(K):
		try:
			resp[:,k] = pi[k]*_pdf(X, mean[k], sigma[k])
		except Exception as e:
			print('error at:', k)
			print(e)
			print('m', mean[k])
			print('sigma', sigma[k])
			input()
	try:
		for n in range(resp.shape[0]):
			s = np.sum(resp[n,:])
			resp[n] /= s
	except Warning:
		print('n:', n)
		print(resp[n])
		input()
	s = np.sum(resp, axis=1)
	if (s==np.nan).any():
		print('dude the sum is very less')

	resp /= s[:, np.newaxis]
	return resp

def _cal_pi(pi, resp):
	pi = np.mean(resp, axis=0)
	return pi

def _cal_mean(mean, X, resp):
	K = mean.shape[0]
	for k in range(K):
		mean[k] = np.sum(X*resp[:,k,np.newaxis], axis=0)/np.sum(resp[:,k])
	return mean

def _cal_sigma(sigma, X, mean, resp, sig_tol=def_sigma_tol, diagonol=False):
	K = mean.shape[0] # number of clusters
	N = X.shape[0] # number of points
	D = X.shape[1] # dimension of the point X[n]
	sigma = np.zeros((K, D, D), dtype=float)
	for k in range(K):
		T = X - mean[k]
		for n in range(N):
			sigma[k] += resp[n,k] * (T[n, :, np.newaxis]*T[n, np.newaxis, :])
		if diagonol:
			sigma[k] = np.diag(np.diag(sigma[k]))
		sigma[k] /= np.sum(resp[:,k])
	sigma = _fix_sigma(sigma, sig_tol)
	return sigma

def _fix_sigma(sigma, tol):
	K = sigma.shape[0]
	for k in range(K):
		sign, q = np.linalg.slogdet(sigma[k])
		if sign == 0 or q<-100:
			sigma[k] = np.diag(np.diag(sigma[k]))
			# print('fixed sigma['+str(k)+']', sigma[k])
		d = np.diag_indices(sigma.shape[1])
		if (sigma[k][d]<tol).any():
			sigma[k][d] = tol

	return sigma

def _pdf(X, mean, sigma):
	return mn.pdf(X, mean, sigma, allow_singular=False)

def _e_step(resp, X, mean, sigma, pi):
	resp = _cal_resp(resp, X, mean, sigma, pi)
	return resp	

def _m_step(mean, sigma, pi, X, resp, sig_tol=def_sigma_tol):
	pi = _cal_pi(pi, resp)
	mean = _cal_mean(mean, X, resp)
	sigma = _cal_sigma(sigma, X, mean, resp, sig_tol)
	return mean, sigma, pi

def _log_likelihood(ll, X, mean, sigma, pi):
	L = np.zeros((mean.shape[0], X.shape[0]), dtype=float)
	# for n in range(X.shape[0]):
	sigma = _fix_sigma(sigma, def_sigma_tol)
	for k in range(mean.shape[0]):
		try:
			L[k] = _pdf(X, mean[k], sigma[k])
		except Exception as e:
			print(e)
			print('ll error')
			# print('error at:', n)
			# print('X', X[n])
			print('k:', k)
			print('m', mean[k])
			print('sigma', sigma[k])
			print(np.linalg.slogdet(sigma[k]))
			# print('L', L[k,n])
			# input()
			input('>> ')		
	L *= pi[:,np.newaxis]
	s = np.sum(L, axis=0)
	# safety for log. input value should not fall below this. (exprimentally found)
	s[s<1e-323]=1e-323
	ll = np.sum(np.log(s))
	return ll

def _assign_mixture(mixture_labels, X, mean, sigma, pi):
	resp = np.ones((X.shape[0], mean.shape[0]), dtype=float)
	resp = _cal_resp(resp, X, mean, sigma, pi)
	mixture_labels = np.ones((X.shape[0]), dtype=int)*-1
	mixture_labels = resp.argmax(axis=1)
	return mixture_labels

def _init(X, mean, diagonol=False):
	km = kmeans(K)
	km.means = mean
	km.assign_clusters(X)
	m, sigma, pi = _extract_from_kmeans(km)
	sigma = _fix_sigma(sigma, def_sigma_tol)
	return sigma, pi

def _extract_from_kmeans(km):
	K = km.n_clusters
	mean = km.means
	D = mean.shape[1] # dimension of the vector space
	clusters = km.clusters
	pi = np.bincount(clusters)/clusters.shape[0]
	sigma = np.ones((K, D, D), dtype=float)
	for k in range(K):
		sigma[k] = np.cov(km.X[clusters==k].T, bias=True)
		# if diagonol:
			# sigma[k] = np.diag(np.diag(sigma[k]))			
	return mean, sigma, pi


def _terminate(old_ll, new_ll, precision):
	if abs(new_ll - old_ll)>precision:
		return False
	else:
		return True

class gmm:
	Name = '' # mane of an instance
	K = 1 # number of mixtures
	s_tol = 1e-3 # min value for variance of a dimension
	mean = np.array([]) # means
	old_means = np.array([])
	sigma = np.array([]) # covarince matrix
	old_sigma = np.array([])
	pi = np.array([]) # mixture coefficients
	old_pi = np.array([])
	resp = np.array([]) # responsibilities
	old_resp = np.array([])
	ll = 0.0 # log likelihood
	old_ll = 0.0
	ll_data = []
	ll_precision = 1e-3

	def __init__(self, n_of_mixtures, sigma_tolerance=1e-3, diagonol=False,
			name='GMM Object'):
		self.K = n_of_mixtures
		self.s_tol = sigma_tolerance
		self.diag = diagonol
		self.name = name

	def init_mean_from_file(self, path_to_file):
		self.old_mean = np.copy(self.mean)
		self.mean = self.read_from_file(path_to_file)

	def init_x_from_file(self, path_to_file):
		self.X = self.read_from_file(path_to_file)

	def read_from_file(self, path_to_file):
		X = np.array([])
		file = Path(path_to_file)
		if file.is_file():
			X = np.loadtxt(path_to_file)
		else:
			print("file", path_to_file,"isn't vaild")
			sys.exit("file", path_to_file,"wasn't valid!")
		return X

	def init_others(self):
		print(self.name, 'Doing init')
		self.old_ll = _log_likelihood(self.old_ll, 
			self.X, self.mean, self.sigma, self.pi)
		self.ll_data.append(self.old_ll)
		self.resp = np.ones((self.X.shape[0], self.mean.shape[0]), dtype=float)
			
	def general_init(self):
		print(self.name, 'Doing init')
		self.sigma, self.pi = _init(self.X, self.mean, self.diag)
		self.init_others()
		
	def init(self, data_file, means_file):
		print(self.name, 'Doing init')
		self.init_x_from_file(data_file)
		self.init_mean_from_file(means_file)

	def do_e_step(self):
		return _e_step(self.resp, self.X, self.mean, self.sigma, self.pi)

	def do_m_step(self):
		return _m_step(self.mean, self.sigma,
			self.pi, self.X, self.resp, self.s_tol)

	def do_em(self):
		print(self.name, 'Doing E step')
		self.resp = self.do_e_step()
		print(self.name, 'Doing M step')
		self.mean, self.sigma, self.pi = self.do_m_step()

	def get_ll(self):
		self.ll = _log_likelihood(self.ll,
			self.X, self.mean, self.sigma, self.pi)

	def get_ll_for_a_point(self, Y):
		self.ll = _log_likelihood(self.ll, Y, self.mean, self.sigma, self.pi)
		return self.ll

	def get_ll_for_points(self, Y):
		res = np.zeros((Y.shape[0]), dtype=float)
		for i in range(res.shape[0]):
			res[i] = self.get_ll_for_a_point(Y[i])
		return res

	def fit(self, data_file, means_file):
		self.init(data_file, means_file)
		self.do_fit()

	def fit_using_kmeans(self, data_file):
		print(self.name, 'Doing a Kmeans first')
		self.init_x_from_file(data_file)
		km = kmeans(self.K)
		km.fit(self.X, self.K, display_progress=True, debug_mode=False, 
			calculate_cov=False, cap=False)
		self.mean, self.sigma, self.pi = _extract_from_kmeans(km)
		print('calling fix sigma')
		self.sigma = _fix_sigma(self.sigma, self.s_tol)
		print('done with fix sigma')
		self.init_others()
		self.do_fit()

	def do_fit(self):
		print('Fitting gmm:', self.name)
		self.ll = self.old_ll + self.ll_precision*1.2
		counter = 0
		while not _terminate(self.ll_data[-1], self.ll, self.ll_precision):
			print(self.name, 'Iteration',counter, 'old:', self.old_ll, 'current:', self.ll)
			self.transfer_data()
			start = timer()
			# print('Doing em')
			self.do_em()
			# print('Saving to intermediate')
			# if self.intermediate != None:
				# self.save_to_folder('intermediate')
			# print('getting log likelihood')
			self.get_ll()
			print(self.name, 'Took', timer()-start, 'seconds.')
			counter+=1

	def transfer_data(self):
		self.transfer_mean()
		self.transfer_pi()
		self.transfer_sigma()
		self.transfer_resp()
		self.transfer_ll()

	def transfer_mean(self):
		self.old_mean = np.copy(self.mean)

	def transfer_pi(self):
		self.old_pi = np.copy(self.pi)

	def transfer_sigma(self):
		self.old_sigma = np.copy(self.sigma)

	def transfer_resp(self):
		self.old_resp = np.copy(self.resp)

	def transfer_ll(self):
		self.old_ll = self.ll
		self.ll_data.append(self.ll)

	def assign_mixture(self, folder):
		X = self.read_from_file(folder+'\\bovw32\\'+folder.split('\\')[-1]
			+'.txt')
		self.mixtures = np.ones((X.shape[0]), dtype=int)*-1
		self.mixtures = _assign_mixture(self.mixtures,
			X, self.mean, self.sigma, self.pi)
		return self.mixtures

	def save_mean(self, output_folder):
		np.savetxt(output_folder+'\\mean.txt', self.mean)

	def save_to_folder(self, output_folder):
		self.save_mean(output_folder)
		np.savetxt(output_folder+'\\resp.txt', self.resp)
		np.savetxt(output_folder+'\\pi.txt', self.pi)
		scipy.io.savemat(output_folder+'\\sigma', mdict={'sigma':self.sigma})
		try:
			np.savetxt(output_folder+'\\log_data.txt', self.ll_data)
		except:
			print(self.name, 'I couldn\'t save the log data.')

	def load_from_folder(self, folder):
		print(self.name, 'Loading from', folder)
		self.mean = np.loadtxt(folder+'\\mean.txt')
		self.pi = np.loadtxt(folder+'\\pi.txt')
		self.sigma = scipy.io.loadmat(folder+'\\sigma')['sigma']
		if len(self.mean.shape)==1:
			d = self.mean.shape[0]
			self.mean = self.mean.reshape((1,d))
			self.pi = self.pi.reshape((1,))

