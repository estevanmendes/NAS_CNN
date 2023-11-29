import tensorflow as tf
import numpy as np
from Pool import get_random_layer
from Utils import output_prints_decorator_factory,default_filenames
import datetime


def check_dimension_compatibility(model:tf.keras.Sequential,layer:tf.keras.layers,pool_of_features,pool_of_features_probability,debug=False) -> tf.keras.layers:
    """
    checks if it is feasible to add the intended layer 
    """
    try:
        ### dumb way of check compatibilty
        model=check_flatten_need(model,layer,debug)
        model.add(layer)
        # model.pop()
        # if added:
        #     model.pop()
                
    except:
        if debug:
            print('Dimension compatibility error')

        layer=get_random_layer(pool_of_features,pool_of_features_probability)
        model=check_dimension_compatibility(model,layer,pool_of_features,pool_of_features_probability)

    if debug:
        print('dimension outcome:',layer)

    return model

def create_model(individual,pool_of_features,pool_of_features_probability,debug=False):
    """
    creates the model indicated by the individual. 
    """
    model=tf.keras.Sequential()
    non_empty_layer=0
    for (index,gene) in enumerate(individual):
        layer_details=pool_of_features[gene]      
                    
        if layer_details['layer'] is not None:
            
            if non_empty_layer==0:
                layer_details['params']['input_shape']=(100,100,3)
                layer=layer_details['layer'](**layer_details['params'])
                non_empty_layer+=1   

            else:
                layer=layer_details['layer'](**layer_details['params'])     
                model=check_dimension_compatibility(model,layer,pool_of_features,pool_of_features_probability,debug=debug)                

            
    layer=tf.keras.layers.Dense(2,activation='softmax')
    model=check_flatten_need(model,layer)
    model.add(layer)
    if debug:
            print('model stack:',*model.layers,sep='\n')

    learning_rate=tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=.1,decay_steps=10000.,decay_rate=0.95)
    opt=tf.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=tf.metrics.mse,metrics=tf.metrics.AUC(name='auc'))
    
    return model

@output_prints_decorator_factory(*default_filenames)
def train_model(trainning_dataset,validation_dataset,model:tf.keras.Sequential,individual,seed=None,verbose=0,max_epochs=20,display=False)-> tf.keras.Sequential:

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3)
    model.fit(trainning_dataset,validation_data=validation_dataset,epochs=max_epochs,verbose=verbose,callbacks=[callback,tensorboard_callback])   

    return model

def evaluate_model(testing_dataset,model:tf.keras.Sequential,verbose=0)->float:
    _,metric=model.evaluate(testing_dataset,verbose=verbose)
    return metric

def check_flatten_need(model:tf.keras.Sequential,layer_to_be_add:tf.keras.layers,debug=False)->tf.keras.Sequential:
    
    """
        Checks if it is required to add a flatten layern, in order of connect dense layers into Convolutional and Maxpooling layers.
    """
    assert 'dense' not in tf.keras.layers.BatchNormalization.__doc__.lower()[:30]
    assert 'dense' not in tf.keras.layers.Conv2D.__doc__.lower()[:30]
    assert 'dense' not in tf.keras.layers.MaxPooling2D.__doc__.lower()[:30]
    assert 'dense' not in tf.keras.layers.GlobalAvgPool2D.__doc__.lower()[:30]
    assert 'dense' in tf.keras.layers.Dense.__doc__.lower()[:30]

    assert 'conv' not in tf.keras.layers.BatchNormalization.__doc__.lower()[:30]
    assert 'conv' in tf.keras.layers.Conv2D.__doc__.lower()[:30]
    assert 'conv' not in tf.keras.layers.MaxPooling2D.__doc__.lower()[:30]
    assert 'conv' not in tf.keras.layers.GlobalAvgPool2D.__doc__.lower()[:30]
    assert 'conv' not in tf.keras.layers.Dense.__doc__.lower()[:30]

    assert 'pool' not in tf.keras.layers.BatchNormalization.__doc__.lower()[:30]
    assert 'pool' not in tf.keras.layers.Conv2D.__doc__.lower()[:30]
    assert 'pool' in tf.keras.layers.MaxPooling2D.__doc__.lower()[:30]
    assert 'pool' in tf.keras.layers.GlobalAvgPool2D.__doc__.lower()[:30]
    assert 'pool' not in tf.keras.layers.Dense.__doc__.lower()[:30]
    
    if debug:
        print('layer to add :',layer_to_be_add)
    layers=model.layers
    if len(layers)>0:
        if 'dense' in layer_to_be_add.__doc__.lower()[:30]:
            for previus_layer in np.flip(layers):
                if 'dense' in previus_layer.__doc__.lower()[:30] or 'flat' in previus_layer.__doc__.lower()[:30] :
                    break
                elif ('conv' in previus_layer.__doc__.lower()[:30] or 'pool' in previus_layer.__doc__.lower()[:30]):
                    model.add(tf.keras.layers.Flatten())
                    return model
    
    return model



def architecture_feasiable(pool_of_features,individual,debug=False):
    """
    creates the model indicated by the individual. 
    """
    model=tf.keras.Sequential()
    non_empty_layer=0
    for (index,gene) in enumerate(individual):
        layer_details=pool_of_features[gene]      
        
            
        if layer_details['layer'] is not None:
            
            if non_empty_layer==0:
                layer_details['params']['input_shape']=(100,100,3)
                layer=layer_details['layer'](**layer_details['params'])
            else:
                layer=layer_details['layer'](**layer_details['params'])
                model=check_flatten_need(model,layer,debug=debug)

            try:            
                model.add(layer)
                non_empty_layer+=1

            except:
                model=None
                return [-1]*len(individual)
                  
    return individual