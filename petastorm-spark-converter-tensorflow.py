# Databricks notebook source
from pyspark.sql.functions import col

from petastorm.spark import SparkDatasetConverter, make_spark_converter

import io
import numpy as np
import tensorflow as tf
from PIL import Image
from petastorm import TransformSpec
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

import horovod.tensorflow.keras as hvd
from sparkdl import HorovodRunner

# COMMAND ----------

IMG_SHAPE = (224, 224, 3)
BATCH_SIZE = 32
NUM_EPOCHS = 5

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. Load data using Spark
# MAGIC ### The flowers dataset

# COMMAND ----------

df = spark.read.format("delta").load("/databricks-datasets/flowers/delta") \
    .select(col("content"), col("label"))

labels = df.select(col("label")).distinct().collect()
label_to_idx = {label: index for index, (label, ) in enumerate(sorted(labels))}
num_classes = len(label_to_idx)
df_train, df_val = df.limit(100).randomSplit([0.9, 0.1], seed=12345)

df_train = df_train.repartition(2)
df_val = df_val.repartition(2)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 2. Cache the Spark DataFrame using Petastorm Spark converter

# COMMAND ----------

# set a cache directory on DBFS FUSE for intermediate data.
spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, 'file:///dbfs/tmp/petastorm/cache')

converter_train = make_spark_converter(df_train)
converter_val = make_spark_converter(df_val)

# COMMAND ----------

print(f"train: {len(converter_train)}, val: {len(converter_val)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feed the data into a single-node TensorFlow models for training
# MAGIC ###Get the model MobileNetV2 from tensorflow.keras

# COMMAND ----------

# First, load the model and inspect the structure of the model
MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet').summary()

# COMMAND ----------

def get_model(lr=0.001):
    # Create the base model from the pre-trained model MobileNet V2
    base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    # Freeze parameters in the feature extraction layers
    base_model.trainable=False
    
    # Add a new classifer layer for trainsfer learning
    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(num_classes)
    
    model = keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])
    
    return model


def get_compiled_model(lr=0.001):
    model = get_model(lr=lr)
    model.compile(optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9),
                 loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])
    return model

# COMMAND ----------

get_compiled_model().summary()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Preprocess images

# COMMAND ----------

def preprocess(content):
    """
    Preprocess an image file bytes for MobileNetV2 (ImageNet)
    """
    image = Image.open(io.BytesIO(content)).resize([224, 224])
    image_array = keras.preprocessing.image.img_to_array(image)
    return preprocess_input(image_array)

def transform_row(pd_batch):
    """
    The input and output of this function are pandas dataframes.
    """
    pd_batch['features'] = pd_batch['content'].map(lambda x: preprocess(x))
    pd_batch['label_index'] = pd_batch['label'].map(lambda x: label_to_idx[x])
    pd_batch = pd_batch.drop(labels=['content', 'label'], axis=1)
    return pd_batch
    
transform_spec_fn = TransformSpec(
    transform_row,
    edit_fields=[('features', np.float32, IMG_SHAPE, False), ('label_index', np.int32, (), False)],
    selected_fields=['features', 'label_index']
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train and evaluate the model on the local machine

# COMMAND ----------

def train_and_evaluate(lr=0.001):
    model = get_compiled_model(lr)
    
    with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, 
                                         batch_size=BATCH_SIZE) as train_dataset, \
         converter_val.make_tf_dataset(transform_spec=transform_spec_fn,
                                      batch_size=BATCH_SIZE) as val_dataset:
        
        train_dataset = train_dataset.map(lambda x: (x.features, x.label_index))
        steps_per_epoch = len(converter_train)//BATCH_SIZE
        
        val_dataset = val_dataset.map(lambda x: (x.features, x.label_index))
        validation_steps = max(1, len(converter_val)//BATCH_SIZE)
        
        
        print(f"step_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")
        
        hist = model.fit(train_dataset, 
                         steps_per_epoch=steps_per_epoch,
                         epochs=NUM_EPOCHS,
                         validation_data=val_dataset,
                         validation_steps=validation_steps,
                         verbose=2)
    return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1]
    
    
loss, accuracy = train_and_evaluate()
print(f"Validation Accuracy:{accuracy}")
        

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4. Feed the data into a distributed hyperparameter tuning function

# COMMAND ----------

def train_fn(lr):
    import tensorflow as tf
    from tensorflow import keras
    loss, accuracy = train_and_evaluate()
    return {'loss': loss, 'status': STATUS_OK}

search_space = hp.loguniform('lr', -10, -4)

argmin = fmin(
    fn=train_fn,
    space=search_space,
    algo=tpe.suggest,
    max_evals=2,
    trials=SparkTrials(parallelism=2))


# COMMAND ----------

argmin

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### 5. Feed the data into a distributed TensorFlow model for training

# COMMAND ----------

def train_and_evaluate_hvd(lr=0.001):
    
    hvd.init()  # Initialize Horovod.
    
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        
    model = get_model(lr)
    
    # Horovod: adjust learning rate based on number of GPUs.
    optimizer = keras.optimizers.SGD(lr=lr * hvd.size(), momentum=0.9)
    dist_optimizer = hvd.DistributedOptimizer(optimizer)
    
    callbacks = [
        # Horovod: broadcast initial variable states from rank 0 to all other processes.
        # This is necessary to ensure consistent initialization of all workers when
        # training is started with random weights or restored from a checkpoint.
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
    ]
  
    # Set experimental_run_tf_function=False in TF 2.x
    model.compile(optimizer=dist_optimizer, 
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=["accuracy"],
                  experimental_run_tf_function=False)
    
    with converter_train.make_tf_dataset(transform_spec=transform_spec_fn, 
                                         cur_shard=hvd.rank(), shard_count=hvd.size(),
                                         batch_size=BATCH_SIZE) as train_dataset, \
         converter_val.make_tf_dataset(transform_spec=transform_spec_fn, 
                                       cur_shard=hvd.rank(), shard_count=hvd.size(),
                                       batch_size=BATCH_SIZE) as val_dataset:
        # tf.keras only accept tuples, not namedtuples
        train_dataset = train_dataset.map(lambda x: (x.features, x.label_index))
        steps_per_epoch = len(converter_train) // (BATCH_SIZE * hvd.size())
 
        val_dataset = val_dataset.map(lambda x: (x.features, x.label_index))
        validation_steps = max(1, len(converter_val) // (BATCH_SIZE * hvd.size()))
    
        hist = model.fit(train_dataset, 
                         steps_per_epoch=steps_per_epoch,
                         epochs=NUM_EPOCHS,
                         validation_data=val_dataset,
                         validation_steps=validation_steps,
                         callbacks=callbacks,
                         verbose=2)
 
    return hist.history['val_loss'][-1], hist.history['val_accuracy'][-1]

# COMMAND ----------

hr = HorovodRunner(np=2)  # This assumes the cluster consists of two workers.
hr.run(train_and_evaluate_hvd)

# COMMAND ----------


