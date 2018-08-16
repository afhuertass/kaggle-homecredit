
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout


drop  = 0.2
def get_model( input_features , nhidden ):

	model = Sequential()


	model.add( Dense( 512 , input_dim = input_features )  )
	model.add( Activation("relu"  , name = "l1")  )
	model.add ( Dropout( 0.2 ) )
	#model.add( BatchNormalization() )
	#Encoder part

	model.add(Dense( 256 ))
	model.add( Activation("relu" , name = "l2")  )
	model.add ( Dropout( 0.2 ) )

	model.add(Dense( 128 ))
	model.add( Activation("relu" , name = "l3")  )
	model.add ( Dropout( 0.2 ) )

	model.add(Dense( 64 ))
	model.add( Activation("relu" , name = "l4")  )
	model.add ( Dropout( 0.2 ) )

	# Begin the decoder parte 
	model.add(Dense( 128 ))
	model.add( Activation("relu" , name = "l5")  )
	model.add ( Dropout( 0.2 ) )

	model.add(Dense( 256 ))
	model.add( Activation("relu" , name = "l6")  )
	model.add ( Dropout( 0.2 ) )

	model.add(Dense( 512 ))
	model.add( Activation("relu" , name = "l7")  )
	model.add ( Dropout( 0.2 ) )

	# output layer
	model.add( Dense( input_features , name = "linear" ) )
	model.add( Activation("linear" ,  name = "output")  )
	#model.add ( Dropout( 0.3 ) )

	return model 



def get_model2( input_features , nhidden  = 1000):

	model = Sequential()


	model.add( Dense( nhidden , input_dim = input_features )  )
	model.add( Activation("relu"  , name = "l1")  )

	#model.add( BatchNormalization() )
	model.add(Dense(nhidden))
	model.add( Activation("relu" , name = "l2")  )
	model.add ( Dropout( 0.2 ) )

	#model.add( BatchNormalization() )
	model.add( Dense(nhidden ) )
	model.add(Activation("relu" , name = "l3") )
	model.add ( Dropout( 0.2 ) )

	#model.add( BatchNormalization() )
	model.add( Dense( nhidden ) )
	model.add(Activation("relu" , name = "l4") )
	model.add ( Dropout( 0.2 ) )

	#model.add( BatchNormalization() )
	model.add( Dense( input_features  , activation="linear")  )
	
	#model.add( Activation("activation='linear'"))
	#model.add( Activation("relu" ,  name = "output")  )
	

	return model 