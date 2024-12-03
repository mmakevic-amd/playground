import os
import tempfile

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_models as tfm

# dataset
tfds_name = "cifar10"
ds, ds_info = tfds.load(tfds_name, with_info=True)
print(ds_info)

# Configure model
exp_config = tfm.core.exp_factory.get_exp_config("resnet_imagenet")
exp_config.task.model.num_classes = 10
exp_config.task.model.input_size = list(ds_info.features["image"].shape)
exp_config.task.model.backbone.resnet.model_id = 50

# Configure training and testing data
batch_size = 128
exp_config.task.train_data.input_path = ""
exp_config.task.train_data.tfds_name = tfds_name
exp_config.task.train_data.tfds_split = "train"
exp_config.task.train_data.global_batch_size = batch_size
exp_config.task.validation_data.input_path = ""
exp_config.task.validation_data.tfds_name = tfds_name
exp_config.task.validation_data.tfds_split = "test"
exp_config.task.validation_data.global_batch_size = batch_size

# adjust the trainer configuration.
device_names = [
    device.name for device in tf.config.list_physical_devices('GPU')
]

print(device_names)

train_steps = 5000
exp_config.trainer.steps_per_loop = 100

exp_config.trainer.summary_interval = 100
exp_config.trainer.checkpoint_interval = train_steps
exp_config.trainer.validation_interval = 1000
exp_config.trainer.validation_steps = ds_info.splits["test"].num_examples // batch_size
exp_config.trainer.train_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.type = "cosine"
exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.1
exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 100

# enable mixed precision
tf.keras.mixed_precision.set_global_policy("mixed_float16")

model_dir = tempfile.mkdtemp()
task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)

model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=None,
    task=task,
    mode="train_and_eval",
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=True,
)

for key, value in eval_logs.items():
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    print(f"{key:20}: {value:.3f}")