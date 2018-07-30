import pandas as pd 
import numpy as np 
import keras 

import os 
random_state = 666

class DataGenerator(object):

	def __init__(self , path , batch_size):

		self.path = path 
		self.data = pd.read_csv(path)
		self.batch_size = batch_size 

		self.nsamples = self.data.shape[ 0 ]
		self.nfeatures = self.data.shape[ 1 ]
		self.nfeatures2permute = int( 0.15* self.nfeatures )

		self.datanp = self.data.values 

	def generate( self ):

		while 1:
			imax = int( self.nsamples / self.batch_size )

			for i in range(imax):

				x , y = self.sample_data( imax )
				yield x , y 

	def getSteps(self):

		return self.nsamples / self.batch_size 

	def getNFeatures(self):

		return self.nfeatures 

	def sample_data( self , indx  ):

		

		d = self.data.sample( self.batch_size , random_state = random_state , replace = True )
		indx_end = min(indx + self.batch_size, self.nsamples )
		cur_len = indx_end - indx
		#print( indx )
		#print(indx_end)

		#print( cur_len )

		rows_to_sample = int( 1 * cur_len)
		x = self.data[ indx : indx_end ]

		# [ batch_size , nfeatures ]
		#print("shappeee ")
	
		
		#print( x.shape )

				# 15 
		cols_to_shuffle = np.random.randint(low=0, high=self.nfeatures ,  size=self.nfeatures2permute  )

		random_rows = np.random.randint(low=0, high=self.nsamples, size = rows_to_sample )
		#random_rows[random_rows>indx] += cur_len

		#print( random_rows )
		#print( cols_to_shuffle )
		#print( noise.shape )
		#print( x.shape )
		noise = x.copy().values
		#print( self.data.iloc[ random_rows , cols_to_shuffle ].shape )
		#print("XXXXXXXXXXXXx")
		noise[ 0:rows_to_sample , cols_to_shuffle   ]  = self.datanp[random_rows[:,None], cols_to_shuffle]
		# lista de 
		#print(noise.iloc[ 0:rows_to_sample , cols_to_shuffle   ] .shape )
		#print("############")
		#rows2select = np.random.choice( self.nfeatures , nfeatures2permute ) 


		#x[ : ,rows2replace ] = x[ : , rows2select]
		return noise , x





