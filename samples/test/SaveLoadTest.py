# 2019/09/07 Hiroyuki Ogasawara
# vim:ts=4 sw=4 et:

import  math
import  numpy as np
import  os
import  sys

#------------------------------------------------------------------------------

class mnist_loader:
    def __init__( self, keras ):
        self.keras= keras

    def convert_x( self, data ):
        data= data.astype( np.float32 )
        data/= 255
        data= np.reshape( data, (data.shape[0], 1,28,28) )
        return  data

    def convert_y( self, label ):
        label= self.keras.utils.to_categorical( label, 10 )
        return  label

    def load( self ):
        (x_train, y_train), (x_test, y_test)= self.keras.datasets.mnist.load_data()
        x_train= self.convert_x( x_train )
        x_test= self.convert_x( x_test )
        y_train= self.convert_y( y_train )
        y_test= self.convert_y( y_test )
        return  (x_train, y_train), (x_test, y_test)


#------------------------------------------------------------------------------

def save_log():
    import tensorflow as tf
    writer= tf.summary.FileWriter( '.' )
    writer.add_graph( tf.get_default_graph() )
    writer.flush()


#------------------------------------------------------------------------------

class ModelConverter:

    def __init__( self ):
        pass

    def keras_to_tf( self, keras_model, file_name, use_predict, use_train, use_freeze ):
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.python.framework import graph_util, graph_io

        graph= tf.get_default_graph()
        s= keras.backend.get_session()
        num= 0
        input_list= []
        for op in keras_model.inputs:
            tensor= graph.get_tensor_by_name( op.name )
            input_list.append( tf.placeholder( tensor.dtype, shape=tensor.shape, name='xinput'+str(num) ) )
            num+= 1
        #for op in keras_model.outputs:
        #    tensor= graph.get_tensor_by_name( op.name )
        print( input_list )

        y= keras_model( *input_list )
        z= tf.identity( y, name='youtput' )

        if not use_train:
            if use_freeze:
                freezed_graph= graph_util.convert_variables_to_constants( s, graph.as_graph_def(), ['youtput'] )
                graph_io.write_graph( freezed_graph, '.', file_name, as_text=False )
            else:
                saver= tf.train.Saver().as_saver_def()
                graph_io.write_graph( graph, '.', file_name, as_text=False )
                s.run( 'save/control_dependency', { 'save/filename:0' : os.path.splitext(file_name)[0] } )

        elif use_train:
            op= keras_model.outputs[0]
            tensor= graph.get_tensor_by_name( op.name )
            y0= tf.placeholder( tensor.dtype, shape=tensor.shape, name='yinput' )
            print( y0 )

            loss= tf.losses.mean_squared_error( labels=y0, predictions=z )
            loss2= tf.identity( loss, name='loss_func' )
            optimizer= tf.train.GradientDescentOptimizer( 0.01 )
            train= optimizer.minimize( loss, name='train' )
            saver= tf.train.Saver().as_saver_def()
            init= tf.global_variables_initializer()
            #with open( file_name, 'wb' ) as fo:
            #    fo.write( graph.as_graph_def().SerializeToString() )
            #graph_io.write_graph( graph, '.', file_name + '.txt', as_text=True )
            graph_io.write_graph( graph, '.', file_name, as_text=False )
            if use_predict:
                s.run( 'save/control_dependency', { 'save/filename:0' : os.path.splitext(file_name)[0] } )


    def keras_to_cntk( self, keras_model, file_name ):
        import cntk
        cfunc= cntk.combine( keras_model.outputs )
        cfunc.save( file_name )


    def load_tensorflow( self, file_name ):
        import tensorflow as tf
        graph= tf.Graph()
        with graph.as_default():
            with open( file_name, 'rb' ) as fi:
                graph_def= tf.GraphDef()
                graph_def.ParseFromString( fi.read() )
                tf.import_graph_def( graph_def, name='' )
        return  graph


    def load_cntk( self, file_name ):
        import cntk
        return  cntk.load_model( file_name )


#------------------------------------------------------------------------------

class MNIST_Test:

    def __init__( self, keras ):
        self.keras= keras


    def build_model( self ):
        layers= self.keras.layers
        models= self.keras.models
        model= models.Sequential()

        model.add( layers.Conv2D( 32, kernel_size=(3,3), padding='same', input_shape=(1,28,28) ) )
        model.add( layers.Activation( 'relu' ) )
        model.add( layers.Conv2D( 64, kernel_size=(3,3) ) )
        model.add( layers.Activation( 'relu' ) )
        model.add( layers.MaxPooling2D( pool_size=(2,2) ) )
        model.add( layers.Flatten() )
        model.add( layers.Dense( 128 ) )
        model.add( layers.Activation( 'relu' ) )
        model.add( layers.Dense( 10 ) );

        model.compile( loss= 'mse', metrics=['accuracy'], optimizer='sgd' )
        model.summary()
        self.model= model


    def train( self ):
        (x_train, y_train), (x_test, y_test)= mnist_loader( self.keras ).load()
        self.model.fit( x_train, y_train, batch_size= 32, epochs= 10, verbose=True )

    def save( self, file_name ):
        self.model.save( file_name )

    def load( self, file_name ):
        self.model= self.keras.models.load_model( file_name )


    def test( self, name, predict_func ):
        batch_size= 128
        count= 0
        (x_train, y_train), (x_test, y_test)= mnist_loader( self.keras ).load()
        data_count= len(x_test)
        loop_count= data_count//batch_size
        for e in range(loop_count):
            shuffle= np.random.randint( data_count, size=batch_size )
            x_data= x_test[shuffle]
            y_data= y_test[shuffle]
            y_result= predict_func( x_data, batch_size )
            for b in range(batch_size):
                if np.argmax(y_result[b]) == np.argmax(y_data[b]):
                    count+= 1
        print( "[[%s]] score=%f" % (name, count*100.0 / (loop_count*batch_size)) )


    def test_keras( self ):
        self.test( 'Keras', lambda x,b: self.model.predict( x, batch_size=b ) )


#------------------------------------------------------------------------------

class ModelTest:

    def __init__( self, backend ):
        self.backend= backend
        if backend == 'cntk':
            os.environ['KERAS_BACKEND']='cntk'
        if backend == 'tensorflow':
            os.environ['KERAS_BACKEND']='tensorflow'
            import tensorflow
            from tensorflow import keras
            self.tf= tensorflow
        else:
            import keras

        self.keras= keras
        self.mnist= MNIST_Test( keras )


    def train( self ):
        self.mnist.build_model()
        self.mnist.train()
        self.mnist.save( 'model_keras_%s.h5' % self.backend )


    def test_keras( self ):
        self.mnist.load( 'model_keras_%s.h5' % self.backend )
        self.mnist.test_keras()


    def dump_op( self, graph, name ):
        for op in graph.get_operations():
            print( op.name, op.type )


    def save_tensorflow( self ):
        converter= ModelConverter()
        graph= self.tf.Graph()
        with graph.as_default():
            self.mnist.load( 'model_keras_tensorflow.h5' )
            converter.keras_to_tf( self.mnist.model, 'model_tensorflow.pb', True, False, False )

        graph= self.tf.Graph()
        with graph.as_default():
            self.mnist.load( 'model_keras_tensorflow.h5' )
            converter.keras_to_tf( self.mnist.model, 'model_tensorflow_freezed.pb', True, False, True )

        graph= self.tf.Graph()
        with graph.as_default():
            self.mnist.load( 'model_keras_tensorflow.h5' )
            converter.keras_to_tf( self.mnist.model, 'model_tensorflow_train.pb', False, True, False )


    def test_tensorflow( self ):
        graph= ModelConverter().load_tensorflow( 'model_tensorflow.pb' )
        with graph.as_default():
            s= self.tf.Session()
            s.run( 'save/restore_all', { 'save/filename:0' : 'model_tf/model' } )
            self.mnist.test( 'TensorFlow', lambda x,b: s.run( 'youtput:0', { 'xinput0:0' : x } ) )

    def test_tensorflow_freezed( self ):
        graph= ModelConverter().load_tensorflow( 'model_tensorflow_freezed.pb' )
        with graph.as_default():
            s= self.tf.Session()
            self.mnist.test( 'TensorFlow FREEZED', lambda x,b: s.run( 'youtput:0', { 'xinput0:0' : x } ) )


    def test_tensorflow_train( self ):
        tf= self.tf
        graph= ModelConverter().load_tensorflow( 'model_tensorflow_train.pb' )
        with graph.as_default():
            s= tf.Session()
            s.run( 'init' )
            batch_size= 32
            (x_train, y_train), (x_test, y_test)= mnist_loader( self.keras ).load()
            data_count= len(x_train)
            for e in range(data_count//batch_size):
                shuffle= np.random.randint( data_count, size=batch_size )
                x_data= x_train[shuffle]
                y_data= y_train[shuffle]
                _,l= s.run( ['train','loss_func:0'], { 'xinput0:0' : x_data, 'yinput:0' : y_data } )
                if e % 100 == 0:
                    print( l )
            self.mnist.test( 'TensorFlow TRAINED', lambda x,b: s.run( 'youtput:0', { 'xinput0:0' : x } ) )



    def save_cntk( self ):
        self.mnist.load( 'model_keras_cntk.h5' )
        converter= ModelConverter()
        converter.keras_to_cntk( self.mnist.model, 'model_cntk.dnn' )


    def test_cntk( self ):
        model= ModelConverter().load_cntk( 'model_cntk.dnn' )
        self.mnist.test( 'CNTK', lambda x,b: model.eval( x ) )


#------------------------------------------------------------------------------

def main():

    if 1:
        model= ModelTest( 'tensorflow' )
        #model.train()
        model.test_keras()

        model.save_tensorflow()
        model.test_tensorflow()
        model.test_tensorflow_freezed()
        model.test_tensorflow_train()


    if 0:
        model= ModelTest( 'cntk' )
        #model.train()
        model.test_keras()
        #model.save_cntk()
        model.test_cntk()


if __name__=='__main__':
    main()

