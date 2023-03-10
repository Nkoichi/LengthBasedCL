#training mlm with the customaized mlm trainer and utility functions

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from scheduler import CLScheduler
import numpy as np
import random
import datetime
import re

from mlm_trainer import train_mlm


#check if GPU is available
print("GPU : ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#random seed
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

#load pretrained tokenizers
tokenizer = RobertaTokenizerFast.from_pretrained(r"wikitext103_tokenizer", max_len=512)


#model
config = RobertaConfig(
    vocab_size=30_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config).to(device)


print("number of parameters: ", model.num_parameters())

#log_name => current time
log_name = re.sub("[- :.]", "_", str(datetime.datetime.today()))

#initialization of the model and optimizer
optimizer=optim.AdamW(model.parameters(), lr=5e-5)
num_steps = 0

#best eval loss
best_eval_loss = 10e5

#curriculum learning
length_based_cl = True

total_cl_steps = 100000 #the number of training steps where pacing function is applied
end_steps = 100000 #the number of steps to end the training

#setting for saving models
override_checkpoint = True #default: True
save_step = 5000



if length_based_cl:
    length_batch_list = [(64, 128), (128, 64)]#[(32, 256), (64, 128), (96, 128), (128, 64)]#[(64, 128), (128, 64), (256, 32), (512, 16)]
    cl_scheduler = CLScheduler(total_steps=total_cl_steps, num_buckets=len(length_batch_list))
    cl_schedule = cl_scheduler.get_schedule("linear", pow=2) #schedule type: linear, root, geom, tangent
    print(f"Length-based CL ({len(length_batch_list)}-stage)")
    print("CL schedule: ", cl_schedule)
else: #Non CL
    length_batch_list = [(128, 64)] #(128, 64)
    cl_schedule = []
    print("Non Curriculum")


for length, batch in length_batch_list:
    print(f"\n\n\n\n************** switch the difficulty level block_size={length} batch_size={batch} **************\n\n\n\n")    
    train_file = r"dataset/tokenized_dataset/wikitext_103_block_%d.txt" % length
    #start training
    model, num_steps, best_eval_loss = train_mlm(
                    model=model,
                    optimizer=optimizer,
                    block_size=length,
                    batch_size=batch,
                    tokenizer=tokenizer,
                    log_dir=r"baseline",
                    train_file=train_file,
                    eval_file=r"dataset/wikitext-103-raw-v1/wikitext-103-raw/wiki.valid.raw",
                    logging_step=1000,
                    cl_schedule=cl_schedule,
                    save_step=save_step,
                    device=device,
                    log_name=log_name,
                    num_steps=num_steps,
                    best_eval_loss=best_eval_loss,
                    override_checkpoint=override_checkpoint,
                    masking_cl=masking_cl,
                    masking_rate=0.15,
                    num_total_steps=end_steps,
                )
    if model == None:
        print("\n\n\nTraining is over.")
        break
