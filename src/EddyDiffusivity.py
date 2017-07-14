import numpy as np


class EddyDiffusivity2D:
	def __init__(self, **kwargs):
	"""
	Parameters:
	gs:      Grid size. Equally spaced in both directions are assumed.
	qlevels: Specify tracer concentration intervals. If there are
	         N numbers in [qlevels], (N-1) intervals are created.
	         Concentration outlies [qlevels] would be ignored during
	         calculation.
	"""
		must_kwargs = ['kappa', 'gs', 'xpts', 'ypts', 'qlevels']

		for kwarg in must_kwargs:
			if kwarg in kwargs:
				self.__dict__[kwarg] = kwargs[kwarg]
				del kwargs[kwarg]
			else:
				raise ValueError('Keyword parameter [%s] is missing.' % (kwarg,))


		for kwarg in kwargs:
			raise ValueError('Unknown parameter [%s].' % (kwarg,))

		if not ( type(self.kappa) is np.ndarray ):
			self.kappa = np.zeros((xpts,ypts)) + self.kappa
		
		self.all_pts = self.xpts * self.ypts

		
	def markAreaByQLevels(self, qfield):
		area_mark = np.zeros(qfield.shape)
		area_mark[:,:] = np.nan 

		for idx, q in np.ndenumerate(qfield):
			for i in range(len(self.qlevels)-1):
				q_l, q_r = self.qlevels[i], self.qlevels[i+1]
				if q_l <= q and q <= q_r:
					area_mark[idx] = i
		return area_mark

	def calGradX(self, field):
		grad = np.zeros(field.shape):
		for (i,j), _ in np.ndenumerate(field):
			l, u = -1, 1
			if i == 0:
				l = 0
			elif i == (field.shape[0] - 1):
				u = 1

			grad[i,j] = (grad[i+u,j] - grad[i+l,j]) / ((u-l) * self.gs)

		return grad

	def calGradY(self, field):
		grad = np.zeros(field.shape):
		for (i,j), _ in np.ndenumerate(field):
			l, u = -1, 1
			if i == 0:
				l = 0
			elif i == (field.shape[0] - 1):
				u = 1

			grad[i,j] = (grad[i,j+u] - grad[i,j+l]) / ((u-l) * self.gs)

		return grad


	def calKappaGradQField(self, qfield):
		return ( self.calGradX(qfield) ** 2.0 + self.calGradY(qfield) ** 2.0 ) * self.kappa

	def calDiffusivity(self, qfield):
		area_mark  = self.markAreaByQLevels(qfield)
		dI_field   = self.calKappaGradQField(qfield)

		dA = np.zeros((len(self.qlevels)-1,))
		dI = dA.copy()
		dQ = self.qlevels[1:] - self.qlevels[0:-1]

		diff = dA.copy()

		for idx, q in np.ndenumerate(qfield):
			i = area_mark[idx]
			dA[i] += 1
			dI[i] += dI_field[idx]

		diff = dA * dI / dQ**2.0 * self.gs**4.0

		return diff


