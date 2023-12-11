from NAS.Utils import *
from NAS.dataset import *
from NAS.GA import *
from NAS.Models import *
from NAS.Pool import *
from NAS.__init__ import *
import argparse

sol_ind_006=[15, 7, 15, 11, 26, 3, 16, 10, 13, 27, 21, 12, 15, 26, 27]
sol_ind_008=[15, 13, 2, 16, 26, 7, 26, 15, 11, 15, 26, 17, 6, 26, 28]
sol_ind_012=[15, 13, 2, 16, 26, 7, 26, 15, 11, 15, 26, 17, 6, 26, 28]
sol_ind_007=[15, 28, 9, 15, 29, 28, 15, 15, 15, 1, 15, 9, 26, 5, 26, 12, 15, 14, 15, 26, 5, 26, 28, 21, 27]
trainning_dataset,validation_dataset,testing_dataset=load_datasets()

for ind in [sol_ind_006,sol_ind_008,sol_ind_012,sol_ind_007]:
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