import tensorflow as tf
import pandas as pd
import os
import numpy as np
import datetime
from deap import base, creator,tools,algorithms
import json
import sys
import random
import time
import aws
import networkx
import matplotlib.pyplot as plt
from simple_ga_algorithms_checkpoint import *
import argparse

np.random.seed(1234)

from typing import Any

default_filenames=['detalhes_arquiteturas_testadas.txt','experimento_log.txt']

def output_prints_decorator_factory(filename_in,filename_out=None):
    def out_prints_decorator(f):
        def wrapper(*args,**kwargs):
            sys.stdout = open(filename_in, '+a')
            results=f(*args,**kwargs)
            if filename_out:
                sys.stdout = open(filename_out, '+a')
            
            return results
        return wrapper
    
    return out_prints_decorator

def load_datasets():
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
    testing_dataset=Dataframe2ImageDataset(testing_df,'path','binary_label_code').create_dataset()
    return training_dataset,validation_dataset,testing_dataset

def individuals(max_depth=15):
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
    return pool_of_features,pool_of_features_probability

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

def architecture_feaseable(pool_of_features,individual,debug=False):
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

def generate_individuals(pool_of_features,pool_of_features_probability,max_depth):
        pool_individuals=np.random.choice(list(pool_of_features.keys()),size=(1000,max_depth),p=pool_of_features_probability)
        pool_individuals_valids=[]
        for ind in pool_individuals:   
            pool_individuals_valids.append(architecture_feaseable(pool_of_features=pool_of_features,individual=ind))

        pool_individuals_valids=np.array(pool_individuals_valids)
        pool_individuals_valids=pool_individuals_valids[np.where(pool_individuals_valids.sum(axis=1)>0)[0]]

        with open(f'arquiteturas_validas_max_depth_{max_depth}.json','+w') as f:
            json.dump(pool_individuals_valids.tolist(),f)

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
    if len(layers)>0:
        if 'dense' in layer_to_be_add.__doc__.lower()[:30]:
            for previus_layer in np.flip(layers):
                if 'dense' in previus_layer.__doc__.lower()[:30] or 'flat' in previus_layer.__doc__.lower()[:30] :
                    break
                elif ('conv' in previus_layer.__doc__.lower()[:30] or 'pool' in previus_layer.__doc__.lower()[:30]):
                    model.add(tf.keras.layers.Flatten())
                    break
    return model

def create_model(pool_of_features,individual,debug=False):
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

def send_results_2_aws(files):
    client=aws.generate_s3_client()  
    bucket='deeplearning-puc'
    for file in files:
        aws.upload_file(client,file,bucket=bucket)

def check_aws_keys():
    
    env1=os.getenv('ACCESS_KEY')
    env2=os.getenv('SECRET_KEY')
    if not (env1 and env2):
        print('\n')
        print('#'*20)
        print('Keys to upload results were not provided')
        print('#'*20)
        print('\n')

def paralelized_trianning():
    pass

def save_logs(id):
    new_filenames=[id+'_'+filename for filename in default_filenames]
    for filename,new_filename in zip(default_filenames,new_filenames):
        os.rename(filename,new_filename)
    
    return new_filenames
    


def main(id,max_depth,generations,population_size,start_gen,num_of_evaluations=1,max_epochs=20,verbose=0):

    @output_prints_decorator_factory(*default_filenames)
    def evaluate(individual,trainning_dataset,validation_dataset,testing_dataset,pool_of_features,fn_no_linear=None,max_epochs=20,num_of_evaluations=1,verbose=0,display=False):
        if display:
            print('model {} is being trainned')
    
        seeds=[1234,345,121,132,234]
        metrics=[]
        for seed in seeds[:num_of_evaluations]:
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
            
            
            if seed==seeds[0]:
                model_raw=create_model(pool_of_features,individual)
                print('\n'*2)
                print(f'individual: {individual}')
                print(model_raw.summary())
                print('\n'*2)

            model=train_model(trainning_dataset,
                            validation_dataset,
                            model_raw,
                            individual,
                            seed=seed,
                            max_epochs=max_epochs,
                            verbose=verbose)
            metrics.append(evaluate_model(testing_dataset,model,verbose=verbose))

        if num_of_evaluations>1:
            metrics_mean=np.mean(metrics)
            metrics_std=np.std(metrics)
            print('\n')
            print(f'individual:{individual}')
            print(f'metrics mean:{metrics_mean:.5f}, std:{metrics_std:.5f}, samples:{len(metrics)}')
            print('\n')
        else:
            metrics_mean=metrics[-1]

        if fn_no_linear!=None:
            metrics_mean=fn_no_linear(metrics_mean)
        
        return metrics_mean,


    check_aws_keys()
    global pool_of_features
    global pool_of_features_probability
    trainning_dataset,validation_dataset,testing_dataset=load_datasets()
    pool_of_features,pool_of_features_probability=individuals(max_depth=max_depth)


    if not os.path.isfile(f'arquiteturas_validas_max_depth_{max_depth}.json'): 
        print('Pool of valid archtectures about to be created')
        generate_individuals(pool_of_features,pool_of_features_probability,max_depth=max_depth)
        



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


    if start_gen!=generations:

        history = tools.History()
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("individual_guess", initIndividual, creator.Individual)
        toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, filename=f"arquiteturas_validas_max_depth_{max_depth}.json",trial_name=id)

        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt,low=0,up=len(pool_of_features), indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate,trainning_dataset=trainning_dataset.batch(10),
                                            validation_dataset=validation_dataset.batch(10),
                                            testing_dataset=testing_dataset.batch(32),
                                            pool_of_features=pool_of_features,
                                            max_epochs=max_epochs,
                                            num_of_evaluations=num_of_evaluations,
                                            fn_no_linear=lambda x: x**3,
                                            verbose=verbose)

        # Decorate the variation operators
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)


    
        population = toolbox.population_guess(pop_size=population_size)
        history.update(population)


        hof = tools.HallOfFame(1)  # salva o melhor individuo que já existiu na pop durante a evolução

        # Gerar as estatísticas
        stats = tools.Statistics(lambda ind:ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)


  

        for gen in range(start_gen,generations+1):
            if gen!=0:
                checkpoint=f'start_gen_1_to_gen_{gen}_checkpoint_name.pkl'
            else:
                checkpoint=None

            pop, log,hof = simple_algorithm_checkpoint(population=population,
                                                    toolbox=toolbox,
                                                    cxpb=0.5,
                                                    mutpb=0.01,
                                                    ngen=generations,
                                                    stats=stats,
                                                    halloffame=hof,
                                                    verbose=True,
                                                    freq=1,
                                                    checkpoint=checkpoint)
    else:
        gen=start_gen
        checkpoint=f'start_gen_0_to_gen_{gen}_checkpoint_name.pkl'
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        generations = cp["generation"]
        hof = cp["halloffame"]
        log = cp["logbook"]

    if gen==generations:
        print('melhor:',hof[0])
        print(create_model(pool_of_features,hof[0],).summary())
        print(evaluate(hof[0],trainning_dataset=trainning_dataset.batch(10),
                                            validation_dataset=validation_dataset.batch(10),
                                            testing_dataset=testing_dataset.batch(32),
                                            pool_of_features=pool_of_features,
                                            num_of_evaluations=5,
                                            ))
        
        # with open(f'id_{id}_individuals_generation.txt','+a') as f:
            # for gen in history.genealogy_history.values():
                # f.write(str(gen)+'\n')
            
        # graph = networkx.DiGraph(history.genealogy_tree)
        # graph = graph.reverse()     # Make the graph top-down
        # colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
        # networkx.draw(graph, node_color=colors)
        # plt.savefig(f'id_{id}_genealogy_tree.png')

        files=[f'id_{id}_individuals_generation.txt',f'arquiteturas_validas_max_depth_{max_depth}.json',f'id_{id}_genealogy_tree.png']
        filename_logs=save_logs(id)
        files.extend(filename_logs)
        send_results_2_aws(files)



if __name__=="__main__":

    parser = argparse.ArgumentParser(description="",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-sg", "--start_gen", help="")
    parser.add_argument("-eg", "--end_gen", help="")
    parser.add_argument("-s", "--steps", help="")

    parser.add_argument("-g", "--gpu",  help="")
    args = parser.parse_args()
    config = vars(args)
    print(config)

    if config['gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
   
    if config['start_gen']:
        start_gen=int(config['start_gen'])
    else:
        start_gen=0

    if config['end_gen']:
        end_gen=int(config['end_gen'])

    elif config['steps']:
        end_gen=start_gen+int(config['steps'])
    else:
        end_gen=30

    testing=False
    id_user='teste_002_'
    global id
    id=id_user#+str(datetime.datetime.now())
    max_depth=15
    generations=end_gen
    population_size=50
    num_of_evaluations=3
    max_epochs=20

    sys.stdout = open(default_filenames[-1], '+a')
    description=f"""
                {start_gen} geração
                experimento de GA
                30 gerações, 50 individuos,3 avaliacoes,15 profundidade maxima,20 maximo epocas
                metrica objetivo: AUC                   
                    
                """
    print(description)
    t1=time.time()
    main(id,
         max_depth=max_depth,
         generations=generations,
         population_size=population_size,
         num_of_evaluations=num_of_evaluations,
         max_epochs=max_epochs,
         start_gen=start_gen
         )
    t2=time.time()
    dt=t2-t1
    print(f'time to run the code : {dt}')