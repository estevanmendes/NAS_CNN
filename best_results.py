from NAS.Utils import *
from NAS.dataset import *
from NAS.GA import *
from NAS.Models import *
from NAS.Pool import *
from NAS.__init__ import *
import argparse

sol_ind_009=[26, 15, 10, 15, 4, 15, 3, 16, 33, 21, 16, 33, 35, 26, 33]
sol_ind_010=[33, 33, 33, 12, 14, 15, 33, 35, 35, 21, 14, 33, 3, 29, 36]
sol_ind_011=[15, 7, 14, 15, 1, 33, 13, 33, 15, 9, 13, 26, 15, 15, 31]

trainning_dataset,validation_dataset,testing_dataset=load_datasets()

for ind in [sol_ind_009,sol_ind_010,sol_ind_011]:
    metric,metric_records=evaluate(ind,trainning_dataset=trainning_dataset.batch(10),
                                                validation_dataset=validation_dataset.batch(10),
                                                testing_dataset=testing_dataset.batch(32),
                                                pool_of_features=pool_of_features,
                                                pool_of_features_probability=pool_of_features_probability,
                                                num_of_evaluations=5,
                                                return_metric_records=True
                                                )
    with open('best_results_analysis.txt','+a') as f:
        f.write(str(ind)+'\n')
        f.write(str(metric)+'\n')
        f.write(str(metric_records)+'\n')


if __name__=="__main__":

    parser = argparse.ArgumentParser(description="",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--gpu",  help="")
    args = parser.parse_args()
    config = vars(args)
    print(config)
    if config['gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'