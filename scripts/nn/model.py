
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras import regularizers
regularizer = 0.000004

dropout = 0.2
def get_model( input_features , nhidden ):

	model = Sequential()


	model.add( Dense( nhidden , input_dim = input_features )  )
	model.add( Activation("relu"  , name = "l1")  )

	model.add( BatchNormalization() )
	model.add(Dense(nhidden , kernel_regularizer=regularizers.l2(regularizer) , kernel_initializer='glorot_normal'  ))
	model.add( Activation("relu" , name = "l2")  )
	model.add ( Dropout( dropout ) )

	model.add( BatchNormalization() )
	model.add( Dense(nhidden  , kernel_regularizer=regularizers.l2(regularizer) , kernel_initializer='glorot_normal' ) )
	model.add(Activation("relu" , name = "l3") )
	model.add ( Dropout( dropout ) )

	model.add( BatchNormalization() )
	model.add( Dense( nhidden , kernel_regularizer=regularizers.l2(regularizer) , kernel_initializer='glorot_normal' ) )
	model.add(Activation("relu" , name = "l4") )
	model.add ( Dropout( dropout ) )

	model.add( BatchNormalization() )
	model.add( Dense( 1  , activation="sigmoid")  )
	
	#model.add( Activation("activation='linear'"))
	#model.add( Activation("relu" ,  name = "output")  )
	

	return model 



