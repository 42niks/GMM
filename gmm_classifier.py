import os
import gmm as G
import numpy as np

class gmm_classifier():
	N = 2 # number of classes
	K = 1 # number of mixtures for 
	gmm = [] # list of gmm objects
	con_mat = np.array([])
	recall = []
	meanrecall = 0.0
	precision = []
	meanprecision = 0.0
	fmeasure = []
	meanfmeasure = 0.0
	accuracy = 0.0


	def __init__(self, N, K):
		self.N = N
		self.K = K
		self.gmm = []
		for i in range(N):
			self.gmm.append(G.gmm(K, name='Class'+str(i)))
		self.con_mat = np.array([])
		self.recall = []
		self.meanrecall = 0.0
		self.precision = []
		self.meanprecision = 0.0
		self.fmeasure = []
		self.meanfmeasure = 0.0
		self.accuracy = 0.0


	def load_from_folders(self, folders):
		if len(folders)!= self.N:
			raise ValueError('Need '+str(self.N)+' folders to intialize from.')
		for i in range(self.N):
			self.gmm[i].load_from_folder(folders[i])

	def save_data(self, folders):
		for i in range(self.N):
			if not os.path.exists(folders[i]):
				os.makedirs(folders[i])
			self.gmm[i].save_to_folder(folders[i])

	def fit(self, data_files, mean_files):
		if len(data_files)!=self.N:
			raise ValueError('Need '+str(self.N)+' data files to intialize from.')
		if len(mean_files)!=self.N:
			raise ValueError('Need '+str(self.N)+' mean files to intialize from.')
		for i in range(self.N):
			self.gmm[i].fit(data_files[i], mean_files[i])

	def fit_data(self, data_files):
		if len(data_files)!=self.N:
			raise ValueError('Need '+str(self.N)+' data files to intialize from.')
		for i in range(self.N):
			self.gmm[i].fit_using_kmeans(data_files[i])		

	# returns one row in a confusion matrix. to be used for
	# the points of whose the class label is already known
	def test_points(self, y):
		t = np.zeros((y.shape[0], 3), dtype=float)
		for i in range(self.N):
			t[:,i] = self.gmm[i].get_ll_for_points(y)
		return np.bincount(t.argmax(axis=1), minlength=self.N).astype(int)

	def test(self, testing_files):
		self.y=[]
		for f in testing_files:
			self.y.append(np.loadtxt(f))
		# p = Pool(processes=3)
		self.con_mat = np.zeros((self.N, self.N), dtype=int)
		for c, y in enumerate(self.y):
			self.con_mat[c] = self.test_points(y)
		self.cal_metrics(self.con_mat)

	def cal_metrics(self, con_mat=None):
		# if con_mat==None:
			# con_mat = self.con_mat
		sumrows = np.sum(con_mat, axis=1)
		sumcols = np.sum(con_mat, axis=0)
		for i in range(self.N):
			self.recall.append(con_mat[i][i]/sumrows[i])
			self.precision.append(con_mat[i][i]/sumcols[i])
			self.fmeasure.append((2*self.precision[0]*self.recall[0])
				/(self.precision[0]+self.recall[0]))
			self.accuracy+=con_mat[i][i]
		self.accuracy = self.accuracy / (sum(sumcols))
		self.meanrecall = np.mean(self.recall)
		self.meanprecision = np.mean(self.precision)
		self.meanfmeasure = np.mean(self.fmeasure)

	def save_metrics(self, dest_folder):
		if not os.path.exists(dest_folder):
			os.makedirs(dest_folder)
		f = open(dest_folder+'\\K'+str(self.K)+'.txt', 'w')
		f.write('Confusion Matrix:\n')
		f.write(str(self.con_mat))
		f.write('\nRecall:\n')
		f.write(str(self.recall))
		f.write('\nMean Recall:\n')
		f.write(str(self.meanrecall))
		f.write('\nPrecision:\n')
		f.write(str(self.precision))
		f.write('\nMean Precision\n')
		f.write(str(self.meanprecision))
		f.write('\nFmeasure:\n')
		f.write(str(self.fmeasure))
		f.write('\nMean Fmeasure:\n')
		f.write(str(self.meanfmeasure))
		f.write('\nAccuracy:\n')
		f.write(str(self.accuracy))
		print('Accuracy is:', self.accuracy)
		f.close()

