import sys
sys.path.append('..')
import aws
import numpy as np
import os
from NAS.__init__ import default_filenames

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

def choice(a,p):
    return np.random.choice(a=a,size=1,p=p)[0]
