import tensorflow as tf 
import pandas as pd
import numpy as np 

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
