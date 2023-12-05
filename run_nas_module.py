import tensorflow as tf
import pandas as pd
import os
import numpy as np
from deap import base, creator,tools,algorithms
import sys
import time
import aws
import networkx
import matplotlib.pyplot as plt
from simple_ga_algorithms_checkpoint import *
import argparse
from NAS.Utils import *
from NAS.dataset import *
from NAS.GA import *
from NAS.Models import *
from NAS.Pool import *
import json

import aws
from NAS import default_filenames,pool_of_features,pool_of_features_probability

def main(id,max_depth,generations,population_size,start_gen,saving_generation,num_of_evaluations=1,max_epochs=20,verbose=0):

    check_aws_keys()
    # global pool_of_features
    # global pool_of_features_probability
    # pool_of_features,pool_of_features_probability=individuals(max_depth=max_depth)

    trainning_dataset,validation_dataset,testing_dataset=load_datasets()

    pool_size=500
    if not os.path.isfile(f'arquiteturas_validas_max_depth_{max_depth}_size_{pool_size}.json'): 
        print('Pool of valid archtectures about to be created')
        generate_individuals(pool_size,pool_of_features,pool_of_features_probability,max_depth=max_depth)
        


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
        toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, filename=f'arquiteturas_validas_max_depth_{max_depth}_size_{pool_size}.json',trial_name=id)

        toolbox.register("mate", tools.cxOnePoint)
        toolbox.register("mutate", tools.mutUniformInt,low=1,up=len(pool_of_features), indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate,trainning_dataset=trainning_dataset.batch(10),
                                            validation_dataset=validation_dataset.batch(10),
                                            testing_dataset=testing_dataset.batch(32),
                                            max_epochs=max_epochs,
                                            num_of_evaluations=num_of_evaluations,
                                            fn_no_linear=lambda x: x**3,
                                            verbose=verbose,
                                            pool_of_features=pool_of_features,
                                            pool_of_features_probability=pool_of_features_probability)
        toolbox.decorate("evaluate", tools.DeltaPenalty(feasiable_model, -10))


        # Decorate the variation operators
        toolbox.decorate("mate", history.decorator)
        toolbox.decorate("mutate", history.decorator)


    
        population = toolbox.population_guess(pop_size=population_size)
        history.update(population)


        hof = tools.HallOfFame(2)  # salva o melhor individuo que já existiu na pop durante a evolução

        # Gerar as estatísticas
        stats = tools.Statistics(lambda ind:ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)


        linear_mutation_prob=np.linspace(0.01,0.4,saving_generation)
        
        for gen in range(start_gen,generations+1):
            if gen==saving_generation:
                break

            elif gen!=0:
                checkpoint=f'start_gen_0_to_gen_{gen-1}_checkpoint_name.pkl'
            else:
                checkpoint=None
            pop, log,hof,genealogy_history = simple_algorithm_checkpoint(population=population,
                                                    toolbox=toolbox,
                                                    cxpb=0.5,
                                                    mutpb=linear_mutation_prob[gen],
                                                    start_gen=gen,
                                                    ngen=gen+1,
                                                    stats=stats,    
                                                    halloffame=hof,
                                                    verbose=True,
                                                    freq=1,
                                                    checkpoint=checkpoint,
                                                    history=history)

    else:
        gen=start_gen
        checkpoint=f'start_gen_0_to_gen_{gen-1}_checkpoint_name.pkl'
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        # population = cp["population"]
        # generations = cp["generation"]
        genealogy_history=cp['genealogy_history']
        hof = cp["halloffame"]
        log = cp["logbook"]

    if gen==saving_generation:
        print('melhor:',hof[0])
        print(create_model(hof[0],pool_of_features,pool_of_features_probability).summary())
        print(evaluate(hof[0],trainning_dataset=trainning_dataset.batch(10),
                                            validation_dataset=validation_dataset.batch(10),
                                            testing_dataset=testing_dataset.batch(32),
                                            pool_of_features=pool_of_features,
                                            pool_of_features_probability=pool_of_features_probability,
                                            num_of_evaluations=5,
                                            ))
        
        with open(f'id_{id}_individuals_generation.txt','+a') as f:
            for gen in genealogy_history.values():
                f.write(str(gen)+'\n')

        with open(f'id_{id}_logbook.txt','+a') as f:
            json.dump(log,f)

        if start_gen==0:
            graph = networkx.DiGraph(history.genealogy_tree)
            graph = graph.reverse()     # Make the graph top-down
            colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
            networkx.draw(graph, node_color=colors)
            plt.savefig(f'id_{id}_genealogy_tree.png')

        # files=[f'id_{id}_individuals_generation.txt',f'arquiteturas_validas_max_depth_{max_depth}.json']#,f'id_{id}_genealogy_tree.png']
        files=[f'arquiteturas_validas_max_depth_{max_depth}_size_{pool_size}.json',f'id_{id}_individuals_generation.txt',f'id_{id}_logbook.txt']
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
        
    saving_generation=30
    testing=False
    id_user='teste_007'
    global id
    id=id_user#+str(datetime.datetime.now())
    max_depth=25
    generations=end_gen
    population_size=50
    num_of_evaluations=3
    max_epochs=1

    sys.stdout = open(default_filenames[-1], '+a')
    description=f"""
                Adam optimizer and linear increase mut prob
                15 layers - only convolutional, pooling, dropout,None,batchnormalization layers
                {start_gen} geração
                experimento de GA
                {start_gen}-{end_gen} gerações, {population_size} individuos,{num_of_evaluations} avaliacoes,{max_depth} profundidade maxima,{max_epochs} maximo epocas
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
         start_gen=start_gen,
         saving_generation=saving_generation
         )
    t2=time.time()
    dt=t2-t1
    print(f'time to run the code : {dt}')