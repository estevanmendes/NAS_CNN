import tensorflow as tf
import numpy as np

img_shape=(100,100,3)
default_filenames=['detalhes_arquiteturas_testadas.txt','experimento_log.txt']
# pool_of_features={1:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':64,'kernel_size':5,'strides':1,'padding':'valid','activation':'relu'}},
#                     2:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':32,'kernel_size':5,'strides':1,'padding':'valid','activation':'relu'}},
#                     3:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':16,'kernel_size':5,'strides':1,'padding':'valid','activation':'relu'}},
#                     4:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':8,'kernel_size':5,'strides':1,'padding':'valid','activation':'relu'}},
#                     5:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':64,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
#                     6:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':32,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
#                     7:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':16,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
#                     8:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':8,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
#                     9:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':8,'kernel_size':3,'strides':1,'padding':'valid','activation':'relu'}},
#                     10:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':64,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
#                     11:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':32,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
#                     12:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':16,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
#                     13:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':8,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
#                     14:{'layer':tf.keras.layers.Conv2D,
#                         'params':{'filters':8,'kernel_size':1,'strides':1,'padding':'valid','activation':'relu'}},
#                     15:{'layer':tf.keras.layers.BatchNormalization,
#                         'params':{}},
#                     16:{'layer':tf.keras.layers.MaxPool2D,
#                         'params':{'pool_size':(2,2)}},
#                     17:{'layer':tf.keras.layers.MaxPool2D,
#                         'params':{'pool_size':(4,4)}},
#                     18:{'layer':tf.keras.layers.MaxPool2D,
#                         'params':{'pool_size':(6,6)}},
#                     19:{'layer':tf.keras.layers.MaxPool2D,
#                         'params':{'pool_size':(8,8)}},
#                     20:{'layer':tf.keras.layers.MaxPool2D,
#                         'params':{'pool_size':(10,10)}},
#                     21:{'layer':tf.keras.layers.AveragePooling2D,
#                         'params':{'pool_size':(2,2)}},
#                     22:{'layer':tf.keras.layers.AveragePooling2D,
#                         'params':{'pool_size':(4,4)}},
#                     23:{'layer':tf.keras.layers.AveragePooling2D,
#                         'params':{'pool_size':(6,6)}},
#                     24:{'layer':tf.keras.layers.AveragePooling2D,
#                         'params':{'pool_size':(10,10)}},
#                     25:{'layer':tf.keras.layers.GlobalMaxPool2D,
#                         'params':{}}, 
#                     26:{'layer':tf.keras.layers.Dense,
#                         'params':{'units':100,'activation':'relu'}},
#                     27:{'layer':tf.keras.layers.Dense,
#                         'params':{'units':80,'activation':'relu'}},
#                     28:{'layer':tf.keras.layers.Dense,
#                         'params':{'units':60,'activation':'relu'}},
#                     29:{'layer':tf.keras.layers.Dense,
#                         'params':{'units':40,'activation':'relu'}}, 
#                     30:{'layer':tf.keras.layers.Dense,
#                         'params':{'units':20,'activation':'relu'}},
#                     31:{'layer':tf.keras.layers.Dense,
#                         'params':{'units':10,'activation':'relu'}},
#                     32:{'layer':tf.keras.layers.Dense,
#                         'params':{'units':5,'activation':'relu'}},
#                     33:{'layer':None,
#                         'params':{}},
#                     34:{'layer':tf.keras.layers.Dropout,
#                         'params':{'rate':0.5}},
#                     35:{'layer':tf.keras.layers.Dropout,
#                         'params':{'rate':0.25}},
#                     36:{'layer':tf.keras.layers.Dropout,
#                         'params':{'rate':0.15}}
                        
#                     }

# pool_of_features_probability=np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,20,3,3,3,3,3,3,3,3,9,3,3,3,3,3,3,3,3,20,2,2,2])
# pool_of_features_probability=pool_of_features_probability/pool_of_features_probability.sum()

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
                    26:{'layer':None,
                        'params':{}},
                    27:{'layer':tf.keras.layers.Dropout,
                        'params':{'rate':0.5}},
                    28:{'layer':tf.keras.layers.Dropout,
                        'params':{'rate':0.25}},
                    29:{'layer':tf.keras.layers.Dropout,
                        'params':{'rate':0.15}}
                        
                    }


pool_of_features_probability=np.array([
                                        3,3,3,3,3,3,3,3,3,3,3,3,3,3,20,3,3,3,3,3,3,3,3,3,3,20,3,3,3,
                                        ])
pool_of_features_probability=pool_of_features_probability/pool_of_features_probability.sum()
