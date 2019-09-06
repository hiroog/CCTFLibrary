# 2019/09/01 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import tensorflow as tf
import numpy as np
import functools
import struct
import os

#------------------------------------------------------------------------------

class mnist_loader:
    def __init__( self ):
        pass

    def convert_x( self, data ):
        data= data.astype( np.float32 )
        data/= 255
        data= np.reshape( data, (data.shape[0], 1,28,28) )
        return  data

    def convert_y( self, keras, label ):
        label= keras.utils.to_categorical( label, 10 )
        return  label

    def load( self ):
        import tensorflow.keras as keras
        (x_train, y_train), (x_test, y_test)= keras.datasets.mnist.load_data()
        x_train= self.convert_x( x_train )
        x_test= self.convert_x( x_test )
        y_train= self.convert_y( keras, y_train )
        y_test= self.convert_y( keras, y_test )
        return  (x_train, y_train), (x_test, y_test)

    def save_image( self, file_name, data ):
        if os.path.exists( file_name ):
            return
        fmt= struct.Struct( '784B' )
        with open( file_name, 'wb' ) as fo:
            fo.write( struct.pack( '>IIII', 0, len(data), 28, 28 ) )
            for xt in data:
                fo.write( fmt.pack( *xt.reshape((28*28,)) ) )

    def save_label( self, file_name, data ):
        if os.path.exists( file_name ):
            return
        fmt= struct.Struct( '1B' )
        with open( file_name, 'wb' ) as fo:
            fo.write( struct.pack( '>II', 0, len(data) ) )
            for xt in data:
                fo.write( fmt.pack( xt ) )

    def save( self ):
        if not os.path.exists( 'data' ):
            os.mkdir( 'data' )
        import tensorflow.keras as keras
        (x_train, y_train), (x_test, y_test)= keras.datasets.mnist.load_data()
        self.save_image( 'data/train-images.idx3-ubyte', x_train )
        self.save_image( 'data/t10k-images.idx3-ubyte', x_test )
        self.save_label( 'data/train-labels.idx1-ubyte', y_train )
        self.save_label( 'data/t10k-labels.idx1-ubyte', y_test )


#------------------------------------------------------------------------------

def get_shape( src ):
    return  [ int(a) for a in src.x.shape if a>0 ]

class Input:
    def __init__( self, size, name= None ):
        self.x= tf.placeholder( tf.float32, shape=[None]+list(size), name=name )

class Dense:
    StaticId= 0
    def __init__( self, src, output_size, name= None ):
        input_size= get_shape(src)[-1]
        self.w= tf.get_variable( 'dense_w'+str(Dense.StaticId), shape=[input_size,output_size], dtype=tf.float32, initializer=tf.initializers.he_normal() )
        self.b= tf.get_variable( 'dense_b'+str(Dense.StaticId), shape=[output_size], dtype=tf.float32, initializer=tf.zeros_initializer() )
        self.x= tf.nn.bias_add( tf.matmul( src.x, self.w ), self.b, name=name )
        Dense.StaticId+= 1

class Conv2d:
    StaticId= 0
    def __init__( self, src, ksize, stride, output_channel, padding='SAME' ):
        input_channel= get_shape(src)[-3]
        self.k= tf.get_variable( 'conv2d_k'+str(Dense.StaticId), shape=[ksize[0],ksize[1],input_channel,output_channel], dtype=tf.float32, initializer=tf.initializers.he_normal() )
        self.b= tf.get_variable( 'conv2d_b'+str(Dense.StaticId), shape=[output_channel], dtype=tf.float32, initializer=tf.zeros_initializer() )
        self.x= tf.nn.bias_add( tf.nn.conv2d( src.x, self.k, strides=[1,1,stride[0],stride[1]], padding=padding, data_format='NCHW' ), self.b, data_format='NCHW' )
        Dense.StaticId+= 1

class MaxPool2d:
    def __init__( self, src, ksize, stride, padding='SAME' ):
        self.x= tf.nn.max_pool( src.x, ksize=[1,1,ksize[0],ksize[1]], strides=[1,1,stride[0],stride[1]], padding=padding, data_format='NCHW' )

class Flatten:
    def __init__( self, src ):
        t= functools.reduce( lambda a,b: a*b, get_shape(src) )
        self.x= tf.reshape( src.x, [-1,t] )

class ReLU:
    def __init__( self, src ):
        self.x= tf.nn.relu( src.x )


#------------------------------------------------------------------------------

class Model_MNIST:
    def __init__( self ):
        pass

    def save_model( self ):

        x0= Input( (1,28,28), 'xinput' )
        y0= Input( (10,), 'yinput' )
        x= Conv2d( x0, (3,3), (1,1), 32 )
        x= ReLU( x )
        x= MaxPool2d( x, (2,2), (2,2) )
        x= Flatten( x )
        x= Dense( x, 128 )
        x= ReLU( x )
        x= Dense( x, 10, 'youtput' )

        loss= tf.losses.mean_squared_error( labels=y0.x, predictions=x.x )
        loss2= tf.identity( loss, name='loss' )
        optimizer= tf.train.GradientDescentOptimizer( 0.01 )
        train= optimizer.minimize( loss, name='train' )

        saver= tf.train.Saver().as_saver_def()
        init= tf.global_variables_initializer()

        with open( 'graph.pb', 'wb' ) as fo:
            fo.write( tf.get_default_graph().as_graph_def().SerializeToString() )


    def load_test( self ):
        graph= tf.Graph()
        with graph.as_default():
            with open( 'graph.pb', 'rb' ) as fi:
                graph_def= tf.GraphDef()
                graph_def.ParseFromString( fi.read() )
                tf.import_graph_def( graph_def, name='' )

        batch_size= 128

        loader= mnist_loader()
        (x_train,y_train),(x_test,y_test)= loader.load()

        s= tf.Session( graph=graph )
        s.run( 'init' )

        for e in range(20): #10
            for i in range(len(x_train)//batch_size):
                shuffle= np.random.randint( len(x_train), size=batch_size )
                x_data= x_train[shuffle]
                y_data= y_train[shuffle]
                _,l= s.run( ['train','loss:0'], { 'xinput:0' : x_data, 'yinput:0' : y_data } )
                if i % 100 == 0:
                    print( 'step', l )

        #s.run( 'save/control_dependency', { 'save/Const:0' : 'tmp/model' } )

        total= 0
        count= len(x_test)//batch_size
        for i in range(count):
            shuffle= np.random.randint( len(x_test), size=batch_size )
            x_data= x_test[shuffle]
            y_data= y_test[shuffle]
            y= s.run( 'youtput:0', { 'xinput:0' : x_data } )
            for b in range(batch_size):
                if np.argmax( y[b] ) == np.argmax( y_data[b] ):
                    total+= 1
        print( 'score=', total *100.0 / (count*batch_size) )


#------------------------------------------------------------------------------

def main():
    mnist_loader().save()
    mnist= Model_MNIST()
    mnist.save_model()
    mnist.load_test()

if __name__=='__main__':
    main()

