import tensorflow as tf
import pandas as pd
import os
import numpy as np
import datetime
from deap import base, creator,tools,algorithms
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


np.random.seed(1234)

from typing import Any

def main():

    class Dataframe2ImageDataset:
        @staticmethod
        def load_image(filepath):
            raw = tf.io.read_file(filepath)        
            tensor = tf.io.decode_image(raw)
            tensor = tf.cast(tensor, tf.float32) / 255.0
            return tensor

        def __init__(self,df,path_column,label_column) -> None:
            self.paths=df[path_column].values
            self.labels=np.eye(2)[df[label_column].values]

        def create_dataset(self):
            dataset = tf.data.Dataset.from_tensor_slices((self.paths,self.labels))
            dataset = dataset.map(lambda filepath, label: (self.load_image(filepath), label))
            # self.dataset=dataset
            return dataset
        
    trainning_df=pd.read_csv('trainning_dataset.csv')
    validation_df=pd.read_csv('validation_dataset.csv')
    testing_df=pd.read_csv('testing_dataset.csv')

    training_dataset=Dataframe2ImageDataset(trainning_df,'path','binary_label_code').create_dataset()
    validation_dataset=Dataframe2ImageDataset(validation_df,'path','binary_label_code').create_dataset()
    testing_df_dataset=Dataframe2ImageDataset(testing_df,'path','binary_label_code').create_dataset()

        
    max_depths=15
    pool_of_features={1:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':64,'kernel_size':5,'strides':1,'padding':'valid','activation':'relu'}},
                    2:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':32,'kernel_size':5,'strides':1,'padding':'valid','activation':'relu'}},
                    3:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':16,'kernel_size':5,'strides':1,'padding':'valid','activation':'relu'}},
                    4:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':8,'kernel_size':5,'strides':1,'padding':'valid','activation':'relu'}},
                    5:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':64,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
                    6:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':32,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
                    7:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':16,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
                    8:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':8,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
                    9:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':8,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
                    10:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':64,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
                    11:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':32,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
                    12:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':16,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
                    13:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':8,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
                    14:{'layer':tf.keras.layers.Conv2D,
                        'params':{'filters':8,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
                    15:{'layer':tf.keras.layers.BatchNormalization,
                        'params':{}},
                    16:{'layer':tf.keras.layers.MaxPool2D,
                        'params':{'pool_size':(2,2)}},
                    17:{'layer':tf.keras.layers.MaxPool2D,
                        'params':{'pool_size':(4,4)}},
                    18:{'layer':tf.keras.layers.MaxPool2D,
                        'params':{'pool_size':(6,6)}},
                    19:{'layer':tf.keras.layers.MaxPool2D,
                        'params':{'pool_size':(8,8)}},
                    20:{'layer':tf.keras.layers.MaxPool2D,
                        'params':{'pool_size':(10,10)}},
                    21:{'layer':tf.keras.layers.AveragePooling2D,
                        'params':{'pool_size':(2,2)}},
                    22:{'layer':tf.keras.layers.AveragePooling2D,
                        'params':{'pool_size':(4,4)}},
                    23:{'layer':tf.keras.layers.AveragePooling2D,
                        'params':{'pool_size':(6,6)}},
                    24:{'layer':tf.keras.layers.AveragePooling2D,
                        'params':{'pool_size':(10,10)}},
                    25:{'layer':tf.keras.layers.GlobalMaxPool2D,
                        'params':{}}, 
                    26:{'layer':tf.keras.layers.Dense,
                        'params':{'units':100,'activation':'relu'}},
                    27:{'layer':tf.keras.layers.Dense,
                        'params':{'units':80,'activation':'relu'}},
                    28:{'layer':tf.keras.layers.Dense,
                        'params':{'units':60,'activation':'relu'}},
                    29:{'layer':tf.keras.layers.Dense,
                        'params':{'units':40,'activation':'relu'}}, 
                    30:{'layer':tf.keras.layers.Dense,
                        'params':{'units':20,'activation':'relu'}},
                    31:{'layer':tf.keras.layers.Dense,
                        'params':{'units':10,'activation':'relu'}},
                    32:{'layer':tf.keras.layers.Dense,
                        'params':{'units':5,'activation':'relu'}},
                    33:{'layer':None,
                        'params':{}},
                    34:{'layer':tf.keras.layers.Dropout,
                        'params':{'rate':0.5}},
                    35:{'layer':tf.keras.layers.Dropout,
                        'params':{'rate':0.25}},
                    36:{'layer':tf.keras.layers.Dropout,
                        'params':{'rate':0.15}}
                        
                    }

    pool_of_features_probability=np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,20,3,3,3,3,3,3,3,3,9,3,3,3,3,3,3,3,3,20,2,2,2])
    pool_of_features_probability=pool_of_features_probability/pool_of_features_probability.sum()

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
        if len(layers)>1:
            if 'dense' in layer_to_be_add.__doc__.lower()[:30]:
                for previus_layer in np.flip(layers):
                    if 'dense' in previus_layer.__doc__.lower()[:30] or 'flat' in previus_layer.__doc__.lower()[:30] :
                        break
                    elif ('conv' in previus_layer.__doc__.lower()[:30] or 'pool' in previus_layer.__doc__.lower()[:30]):
                        model.add(tf.keras.layers.Flatten())
                        break
        return model

    def architecture_feaseable(individual,debug=False):
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
                    model=check_flatten_need(model,layer,debug=debug)
                    layer=layer_details['layer'](**layer_details['params'])
                
                try:            
                    model.add(layer)
                except ValueError:
                    model=None
                    break

        if model is None:
            return [-1]*len(individual)
        else:
            return individual

    if not os.path.isfile('arquiteturas_validas.json'):    
        pool_individuals=np.random.choice(list(pool_of_features.keys()),size=(1000,max_depths),p=pool_of_features_probability)
        pool_individuals_valids=[]
        for ind in pool_individuals:   
            pool_individuals_valids.append(architecture_feaseable(ind))

        pool_individuals_valids=np.array(pool_individuals_valids)
        pool_individuals_valids=pool_individuals_valids[np.where(pool_individuals_valids.sum(axis=1)>0)[0]]

        with open('arquiteturas_validas.json','+w') as f:
            json.dump(pool_individuals_valids.tolist(),f)


    space_checked={}

    def get_random_layer()->tf.keras.layers:
        """ selects one random layer from the pool of features"""
        layer_index=np.random.choice(list(pool_of_features.keys()),1,p=pool_of_features_probability)[0]
        layer_details=pool_of_features[layer_index]
        if layer_details['layer'] is not None:
            layer=layer_details['layer'](**layer_details['params'])
        else:
            layer=get_random_layer()
            
        
        return layer

    def check_dimension_compatibility(model:tf.keras.Sequential,layer:tf.keras.layers,debug=False) -> tf.keras.layers:
        """
        checks if it is feasible to add the intended layer 
        """
        try:
            ### dumb way of check compatibilty
            model.add(layer)
            model.pop()
            
        except ValueError:
            if debug:
                print('Dimension compatibility error')

            layer=get_random_layer()
            layer=check_dimension_compatibility(model,layer)

        if debug:
            print('dimension outcome:',layer)

        return layer

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
        if len(layers)>1:
            if 'dense' in layer_to_be_add.__doc__.lower()[:30]:
                for previus_layer in np.flip(layers):
                    if 'dense' in previus_layer.__doc__.lower()[:30] or 'flat' in previus_layer.__doc__.lower()[:30] :
                        break
                    elif ('conv' in previus_layer.__doc__.lower()[:30] or 'pool' in previus_layer.__doc__.lower()[:30]):
                        model.add(tf.keras.layers.Flatten())
                        break
        return model

    def create_model(individual,debug=False):
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
                    layer=layer_details['layer'](**layer_details['params'])
                    layer=check_dimension_compatibility(model,layer,debug=debug)
                
                model.add(layer)
                non_empty_layer+=1   

                
        layer=tf.keras.layers.Dense(2,activation='softmax')
        model=check_flatten_need(model,layer)
        model.add(layer)
        if debug:
                print('model stack:',*model.layers,sep='\n')

        learning_rate=tf.optimizers.schedules.ExponentialDecay(initial_learning_rate=.1,decay_steps=10000.,decay_rate=0.95)
        opt=tf.optimizers.SGD(learning_rate=learning_rate)
        model.compile(optimizer=opt, loss=tf.metrics.mse,metrics=tf.metrics.AUC(name='auc'))
        
        return model

    def train_model(model:tf.keras.Sequential,individual,verbose=0)-> tf.keras.Sequential:
        if str(individual) not in space_checked.keys():
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3)
            model.fit(training_dataset.batch(10),validation_data=validation_dataset.batch(10),epochs=20,verbose=verbose,callbacks=[callback,tensorboard_callback])   
            space_checked[str(individual)]=model
        else:
            model=space_checked[str(individual)]

        return model

    def evaluate_model(model:tf.keras.Sequential,verbose=0)->float:
        _,metric=model.evaluate(testing_df_dataset.batch(32),verbose=verbose)
        return metric



    def evaluate(individual,num_of_evaluations=1,verbose=0):
        seeds=[1234,345,121,132,234]
        metrics=[]
        for seed in seeds[:num_of_evaluations]:
            model=create_model(individual)
            model=train_model(model,individual,verbose=verbose)
            metrics.append(evaluate_model(model,verbose=verbose))
        metrics=np.mean(metrics)
        
        return metrics,

    if testing:
        individual=[12,#conv
                    16,#pooling
                    15,#norm
                    33,#empty
                    34,#dropout
                    1,#conv
                    2,#conv
                    3,#conv
                    33#empty
                    ]
        model=create_model(individual,debug=True)
        model=train_model(model,individual,verbose=1)
        evaluate_model(model)


    def choice(a,p):
       return np.random.choice(a=a,size=1,p=p)[0]


    def initIndividual(icls, content):
        return icls(content)

    def initPopulation(pcls, ind_init,pop_size,trial_name, filename):
        with open(filename, "r") as pop_file:
            contents = np.array(json.load(pop_file))
        # contents=np.array(contents)
        index_ind_selected=np.random.choice(np.arange(0,len(contents)),size=pop_size,replace=False)
        pop=contents[index_ind_selected,:]

        with open(trial_name+'_population_selected.json','w') as f:
            json.dump(pop.tolist(),f)


        return pcls(ind_init(c) for c in pop)

    history = tools.History()



    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # toolbox.register("attribute", choice,a=list(pool_of_features.keys()),p=pool_of_features_probability)
    # toolbox.register("individual", tools.initRepeat, creator.Individual,toolbox.attribute, n=15)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("individual_guess", initIndividual, creator.Individual)
    toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, filename="arquiteturas_validas.json",trial_name='001')

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutUniformInt,low=0,up=len(pool_of_features), indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate,num_of_evaluations=1,verbose=1)

    # Decorate the variation operators
    toolbox.decorate("mate", history.decorator)
    toolbox.decorate("mutate", history.decorator)


    population_size=5
    # pop=toolbox.population(n=population_size)
    population = toolbox.population_guess(pop_size=population_size)


    hof = tools.HallOfFame(1)  # salva o melhor individuo que já existiu na pop durante a evolução

    # Gerar as estatísticas
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    stats.register('avg', np.mean)
    stats.register('std', np.std)
    stats.register('min', np.min)
    stats.register('max', np.max)


    generations=30
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.01, ngen=generations, stats=stats, halloffame=hof, verbose=True)

    print('melhor:',hof[0])
    print(create_model(hof[0]).summary())
    print(evaluate(hof[0]))


    with open(f'experiment_gen_{generations}_pop_{population_size}_{datetime.datetime.now()}.json','w+') as f:
        json.dump(log,f)


if __name__=="__main__":
    print('starting')
    testing=False
    main()