<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Code to finetune CLIP model on our own dataset

The following example showcases how to train a CLIP model using a pre-trained vision and text encoder based on your own dataset.

Such a model can be used for natural language image search and potentially zero-shot image classification.
The model is inspired by [CLIP](https://openai.com/blog/clip/), introduced by Alec Radford et al.
The idea is to train a vision encoder and a text encoder jointly to project the representation of images and their
captions into the same embedding space, such that the caption embeddings are located near the embeddings
of the images they describe. In this way, the model can help map text descriptions of images to images.

### Prepare your dataset
We recommend that the dataset should be in the form of .csv file. In the csv file, there should be two columns. The first column, with column name "image_path", stores the paths to your image data. The second column, with the column name "caption", stores the label of each image

### Create a model from a vision encoder model and a text encoder model，tokenizer，and image processor
Next, we creat a model, which is CLIP. It is a pretrained model and Huggingface has provided flexible and easy access to different structures of it. In this script, we use the model from this link: [clip-vit-base-patch32 ](https://huggingface.co/openai/clip-vit-base-patch32) 
Tokenizer is going to process the text to the form that model can use. Image processor is used to preprocess the images to the form the model can use. The link also provides the image processor and tokenizer suitable to the model. Huggingface also provides methods to directly import them.
```python3
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
```
We apply the code directly in the script because at present we have not achieved the process of testing different structure of CLIP. If you want to change model, you can adjust it in the script or make is become a argument.

### Train the model
before run the run_clip.py. Confirm you have the right format of dataset and you have successfully imported model, tokenizer, and image processor. This bunch of arguments are just used to test if we can run the run_clip.py successfully. Finally, we can run the example script to train the model :

```bash
python3 run_clip.py   --output_dir /home/xz306/Data-Climate-and-AI/src/fine_tuning/try/clip-roberta-finetuned  --train_file /home/xz306/transformers/data/powerplanttrain.csv   --image_column image_path --caption_column caption   --remove_unused_columns=False   --do_train    --per_device_train_batch_size="8" --per_device_eval_batch_size="8" --learning_rate="5e-5"   --warmup_steps="0"   --weight_decay 0.1  --max_steps 10  --overwrite_output_dir --logging_steps 1
```
output_dir is where to save your tuned model. If you do everything correctly, there should be many files generated under the output dir, including the checkpoints. A folder called wandb should also be generated under your current path which contains the finetuning records and log files. If you are going to test the performance of tuned model, the procedure is same as how to import model, tokenizer, and image processor, but change the link to your own output_dir.


We also deployed the LoRA algorithm in the script to improve the efficiency of model training. To use it, change the part of script that imports model to the code below.
```python3
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["k_proj","v_proj"],
    lora_dropout=0.1,
    bias="none",
)
config = model.config
model = get_peft_model(model, peft_config)
```



# More arguments 
### Model arguments
model_name_or_path: The name of the pretrained model or the path where the model is stored. This could be a model identifier from Hugging Face's model hub. Examples: "openai/clip-vit-base-patch32", "./my_pretrained_clip_model"

config_name: The name of the pretrained configuration or its path if it's not the same as model_name_or_path. This is optional, if not provided, model_name_or_path will be used.

tokenizer_name: The name or path of the pretrained tokenizer if it's not the same as model_name_or_path. This is optional, if not provided, model_name_or_path will be used.

image_processor_name: The name or path of the preprocessor configuration to be used for images. This would be the name of the preprocessor used for images. Examples could include standard preprocessors available in PyTorch or TensorFlow, or a custom one you've developed. Example: "torchvision.transforms.ToTensor", "./my_custom_image_processor"

cache_dir: The directory where you want to store the pretrained models downloaded from Hugging Face's S3 storage. This is optional.

model_revision: The specific version of the model to use. This could be a branch name, tag name, or commit id in the model repository. The default is "main".

use_fast_tokenizer: This is a boolean flag determining whether to use one of the fast tokenizers (backed by the tokenizers library) or not. The default is True.

use_auth_token: This is a boolean flag determining whether to use the token generated when running huggingface-cli login. This is necessary if you want to use this script with private models. The default is False.

freeze_vision_model: This is a boolean flag determining whether to freeze the vision model parameters or not during training. If set to True, the vision model parameters will not be updated during training. The default is False.

freeze_text_model: This is a boolean flag determining whether to freeze the text model parameters or not during training. If set to True, the text model parameters will not be updated during training. The default is False


### Data arguments
dataset_name: Name of the dataset to use. It refers to datasets in the Hugging Face's datasets library. Example inputs: "coco", "imagenet".

dataset_config_name: Configuration of the dataset to use if the dataset has different versions or configurations. Example inputs: "2017" for the COCO dataset.

data_dir: The directory where the dataset is stored. This is used when data is locally stored. Example inputs: "/home/user/dataset".

image_column: The name of the column in the dataset that contains the image file paths. Example inputs: "image_path", "images".

caption_column: The name of the column in the dataset that contains the image captions. Example inputs: "caption", "labels".

train_file: The path to the training data file (a jsonlines file, json or csv). Example inputs: "/path/to/train.json".

validation_file: The path to the evaluation data file (a jsonlines file, json or csv). Example inputs: "/path/to/val.json".

test_file: The path to the testing data file (a jsonlines file, json or csv). Example inputs: "/path/to/test.json".

max_seq_length: Maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. Example inputs: 128, 512.

max_train_samples: For debugging or quicker training, truncate the number of training examples to this value if set. Example inputs: 5000, 10000.

max_eval_samples: For debugging or quicker training, truncate the number of evaluation examples to this value if set. Example inputs: 1000, 5000.

overwrite_cache: Overwrite the cached training and evaluation sets. As this is a boolean, the options are True or False. Se it to true when change the content of the same dataset.

preprocessing_num_workers: The number of processes to use for the preprocessing. Example inputs: 4, 8.

Note that the __post_init__ function is validating that either a dataset_name is provided or train_file and validation_file are provided. It also checks that these files, if provided, are of type "csv" or "json".

### Training arguments. 
Move to: https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer#transformers.TrainingArguments
