
import numpy as np
from anomalib.data.utils import ValSplitMode
from torchvision.transforms.v2.functional import to_pil_image
from anomalib.data.image.folder import Folder
from anomalib import TaskType
import os
import torch
from anomalib.models import ReverseDistillation
from anomalib.loggers import AnomalibWandbLogger
from anomalib.engine import Engine    
from anomalib.deploy import ExportType
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    

# from utils import show_image_list

# set the dataset root for a particular category
dataset_root = "/N/u/aw133/Quartz/Desktop/projects/anomalib-detection/dataset/tissue_images/kidney_tissue"



# Create the datamodule
datamodule = Folder(
    name="kidney",
    root=dataset_root,
    normal_dir="normal",#directory with normal images
    abnormal_dir="abnormal",#directory with anomaly images
    task=TaskType.CLASSIFICATION,
    seed=42,#every time, repeat the same data split
    normal_split_ratio=0.2, # default value
    val_split_mode=ValSplitMode.FROM_TEST, # default value
    val_split_ratio=0.5, # default value
    train_batch_size=2, # default value
    eval_batch_size=32, # default value
    image_size=(512,512),
    num_workers=16
)

# Setup the datamodule
datamodule.setup()


# Train images
i, data_train = next(enumerate(datamodule.train_dataloader()))#iterate over batches of data, data batch is assigned to variable data_train
print(data_train.keys(), data_train["image"].shape) # retrieve shape of the tensor
# for each key extract the first image
print("data_train['image_path'][0]: {} - data_train['image'][0].shape: {} - data_train['label'][0]: {} - torch.max(data_train['image][0]): {} - torch.min(data_train['image][0]): {}".format(data_train['image_path'][0], data_train['image'][0].shape, data_train['label'][0], torch.max(data_train['image'][0]), torch.min(data_train['image'][0])))

img_train = to_pil_image(data_train["image"][0].clone())

# val images
i, data_val = next(enumerate(datamodule.val_dataloader()))
# for each key extract the first image
print("data_val['image_path'][0]: {} - data_val['image'][0].shape: {} - data_val['label'][0]: {}".format(data_val['image_path'][0], data_val['image'][0].shape, data_val['label'][0]))
img_val = to_pil_image(data_val["image"][0].clone())

# test images
i, data_test = next(enumerate(datamodule.test_dataloader()))
# for each key extract the first image
print("data_test['image_path'][0]: {} - data_test['image'][0].shape: {} - data_test['label'][0]: {}".format(data_test['image_path'][0], data_test['image'][0].shape, data_test['label'][0]))
img_test = to_pil_image(data_test["image"][0].clone())

# from the datamodule extract the train, val and test Pandas dataset and collect all the info in a csv
train_dataset = datamodule.train_data.samples
test_dataset = datamodule.test_data.samples
val_dataset = datamodule.val_data.samples

# check the data distribution for each category in each data split
print("TRAIN DATASET FEATURES")
print(train_dataset.info())
print("")
print("IMAGE DISTRIBUTION BY CLASS")
print("")
desc_grouped = train_dataset[['label']].value_counts()
print(desc_grouped)
print("----------------------------------------------------------")
print("TEST DATASET FEATURES")
print(test_dataset.info())
print("")
print("IMAGE DISTRIBUTION BY CLASS")
print("")
desc_grouped = test_dataset[['label']].value_counts()
print(desc_grouped)
print("----------------------------------------------------------")
print("VAL DATASET FEATURES")
print(val_dataset.info())
print("")
print("IMAGE DISTRIBUTION BY CLASS")
print("")
desc_grouped = val_dataset[['label']].value_counts()
print(desc_grouped)

datamodule.train_data.samples.to_csv(os.path.join("/N/u/aw133/Quartz/Desktop/projects/anomalib-detection/data", "datamodule_train.csv"), index=False)
datamodule.test_data.samples.to_csv(os.path.join("/N/u/aw133/Quartz/Desktop/projects/anomalib-detection/data", "datamodule_test.csv"), index=False)
datamodule.val_data.samples.to_csv(os.path.join("/N/u/aw133/Quartz/Desktop/projects/anomalib-detection/data", "datamodule_val.csv"), index=False)




# 1 - instantiate the model  
model = ReverseDistillation()
callbacks = [
    ModelCheckpoint(
        mode="max",
        monitor="image_AUROC",
        save_last=True,
        verbose=True,
        auto_insert_metric_name=True,
        every_n_epochs=1,
    ),
    EarlyStopping(
        monitor="image_AUROC",
        mode="max",
        patience=0,#Number of epochs with no improvement after which training will be stopped, defalt
    ),
]



#instantiate the engine
engine = Engine(
    max_epochs=1000,
   callbacks=callbacks,
    pixel_metrics="AUROC",
    accelerator="auto",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
    devices=1,
    task=TaskType.CLASSIFICATION,
    log_every_n_steps=1
)

# fit
print("Fit...")
engine.fit(datamodule=datamodule, model=model)

# test
print("Test...")
engine.test(datamodule=datamodule, model=model)

# 7 - export torch weights
print("Export weights...")
path_export_weights = engine.export(export_type=ExportType.TORCH,
                                    model=model)

print("path_export_weights: ", path_export_weights)

