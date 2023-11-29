from NAS.Models import *
from NAS.Utils import output_prints_decorator_factory
import numpy as np
from NAS import pool_of_features,default_filenames
import json
import random
import tensorflow as tf

def feasiable_model(individual):
    """
    delta penalty decorator does not accept args and kwargs for feasible function. Then, global variable will be used as a workaround
    """
    result=architecture_feasiable(pool_of_features=pool_of_features,individual=individual)
    if -1 in result:
        return False
    else:

        return True

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

@output_prints_decorator_factory(*default_filenames)
def evaluate(individual,trainning_dataset,validation_dataset,testing_dataset,pool_of_features,pool_of_features_probability,fn_no_linear=None,max_epochs=20,num_of_evaluations=1,verbose=0,display=False):
        if display:
            print('model {} is being trainned')
    
        seeds=[1234,345,121,132,234]
        metrics=[]
        for seed in seeds[:num_of_evaluations]:
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
            
            
            if seed==seeds[0]:
                model_raw=create_model(individual,pool_of_features,pool_of_features_probability,debug=True)
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

