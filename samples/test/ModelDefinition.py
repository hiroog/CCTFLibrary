# 2019/09/07 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.python.framework import graph_io


x0= layers.Input( shape=(1,28,28), name='xinput0' )
y0= layers.Input( shape=(10,), name='yinput' )
x= layers.Conv2D( 32, kernel_size=(3,3), padding='same', input_shape=(1,28,28) )( x0 )
x= layers.Activation( 'relu' )( x )
x= layers.Conv2D( 64, kernel_size=(3,3) )( x )
x= layers.Activation( 'relu' )( x )
x= layers.MaxPooling2D( pool_size=(2,2) )( x )
x= layers.Flatten()( x )
x= layers.Dense( 128 )( x )
x= layers.Dropout( 0.5 )( x )
x= layers.Activation( 'relu' )( x )
x= layers.Dense( 10 )( x );
x= tf.identity( x, name='youtput' )

loss_func0= tf.losses.mean_squared_error( labels=y0, predictions=x )
loss_func= tf.identity( loss_func0, name='loss_func' )

optimizer= tf.train.AdamOptimizer( 0.001 )

trainer= optimizer.minimize( loss_func, name='train' )
saver= tf.train.Saver().as_saver_def()
init= tf.global_variables_initializer()

graph_io.write_graph( tf.get_default_graph(), '.', 'model.pb', as_text=False )
graph_io.write_graph( tf.get_default_graph(), '.', 'model.pb.txt', as_text=True )

#writer= tf.summary.FileWriter( '.' )
#writer.add_graph( tf.get_default_graph() )
#writer.flush()

