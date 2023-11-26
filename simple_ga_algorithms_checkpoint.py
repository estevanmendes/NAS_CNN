import pickle
import random
from deap import tools,algorithms
def _simple_algorithm_checkpoint(start_gen,logbook,toolbox,cxpb,mutpb,freq,ngen,stats,population=None,halloffame=None,checkpoint=None,verbose=False):
    for gen in range(start_gen, ngen):
        
            population = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            halloffame.update(population)
            record = stats.compile(population)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            population = toolbox.select(population, k=len(population))

            if (gen +1) % freq == 0:
                # Fill the dictionary using the dict(key=value[, ...]) constructor
                cp = dict(population=population, generation=gen, halloffame=halloffame,
                        logbook=logbook, rndstate=random.getstate())

                with open(f"start_gen_{start_gen+1}_to_gen_{gen+1}_checkpoint_name.pkl", "wb") as cp_file:
                    pickle.dump(cp, cp_file)
        
    
    return population,logbook



def simple_algorithm_checkpoint(toolbox,cxpb,mutpb,freq,ngen,stats,population=None,halloffame=None,checkpoint=None,verbose=False):
    if checkpoint:
        # A file name has been given, then load the data from the file
        with open(checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        starting_population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        population,logbook=_simple_algorithm_checkpoint(start_gen=start_gen,
                                                        logbook=logbook,
                                                        toolbox=toolbox,
                                                        cxpb=cxpb,
                                                        mutpb=mutpb,
                                                        freq=freq,
                                                        ngen=ngen,
                                                        stats=stats,
                                                        population=starting_population,
                                                        halloffame=halloffame,
                                                        checkpoint=checkpoint,
                                                        verbose=verbose)
        starting_population.extend(population)
        population=starting_population
    
    else:
        start_gen = 0
        logbook = tools.Logbook()
        population,logbook=_simple_algorithm_checkpoint(start_gen=start_gen,
                                                        logbook=logbook,
                                                        toolbox=toolbox,
                                                        cxpb=cxpb,
                                                        mutpb=mutpb,
                                                        freq=freq,
                                                        ngen=ngen,
                                                        stats=stats,
                                                        population=population,
                                                        halloffame=halloffame,
                                                        checkpoint=checkpoint,
                                                        verbose=verbose)              
    return population, logbook