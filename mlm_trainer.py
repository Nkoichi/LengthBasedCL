#My custom Trainer class
import torch
from tqdm import tqdm
import datetime
import os
import time
from utils import mask_token, time_to_loss_score, preprocess_inputs, get_lr_scheduler, masking_rate_linear
from mlm_evaluator import evaluate_mlm
from dataset import MLMDataset
from torch.utils.data import Dataset, DataLoader



def train_mlm(
    model,
    optimizer,
    block_size,
    batch_size,
    tokenizer,
    train_file,
    eval_file,
    log_dir,
    logging_step,
    cl_schedule,
    scheduler_function,
    save_step,
    device,
    log_name,
    num_steps,
    best_eval_loss,
    override_checkpoint,
    masking_cl,
    masking_rate=0.15,
    num_total_steps=100000,
    baby_step=False,
    baby_step_patience=10,
    ):


    #patience counter for baby step
    patience_counter=0

    #datasets
    print(f" *********** loading datasets ({block_size}) *********** ")
    train_dataset = MLMDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=block_size,
    )

    #dataloader
    dataloader = DataLoader(train_dataset, batch_size=batch_size)

    #lr_scheduler (linear)
    #lr_scheduler = get_lr_scheduler(optimizer, num_warmup_steps=0, num_training_steps=num_epochs)

    #initialization
    running_loss = 0.0
    running_corrects = 0
    num_total_masks = 0
    previous_tradeoff_score = 0
    training_time = 0.0
    if masking_cl:
        max_masking_rate = masking_rate
    #loop (epoch)
    while True:
        #training mode
        model.train()
        #vocab schedule steps = lamb (the ratio of available data) * total steps per epoch
        if scheduler_function != None:
            vocab_schedule_steps = round(scheduler_function(num_steps) * len(dataloader))
        else:
            vocab_schedule_steps = None

        #loop (data sampling)
        for sample_id, inputs in tqdm(enumerate(dataloader)):
            #preprocess inputs
            try:
                inputs = preprocess_inputs(inputs, tokenizer)
            except ValueError:
                print("---- ValueError----")
                continue
            #end of a batch processing
            start_time = time.time()
            num_steps += 1

            #the limitation of dataset in vocab CL
            #if sample_id > vocab_limit:
                #continue

            #randomly mask tokens and get labels
            if masking_cl:
                masking_rate = masking_rate_linear(num_steps, num_total_steps, start_rate=0.05, max_rate=max_masking_rate, decrease=True)
            inputs, labels, masked_positions = mask_token(inputs, tokenizer, masking_rate)

            #move all data to CPU or GPU memory
            inputs = inputs.to(device)
            labels = labels.to(device)
            masked_positions = masked_positions.to(device)

            #initialize gradients
            optimizer.zero_grad()

            #calculate loss
            with torch.set_grad_enabled(True):
                outputs = model(inputs, labels=labels)
                loss = outputs.loss
                running_loss += loss.item()
                #backprop
                loss.backward()
                optimizer.step()

            masked_logits = outputs.logits.index_select(1, masked_positions) #outputs => batch, len(masked_positions), vocab
            pred_ids = torch.max(masked_logits, 2).indices #batch, len(masked_positions)
            target_ids = labels.index_select(1, masked_positions)

            running_corrects += torch.sum(pred_ids==target_ids)
            num_total_masks += labels.size(0) * labels.size(1) #batch * number of masks
            #end of a batch processing
            training_time += time.time() - start_time
            #show loss and mlm accuracy
            if num_steps % logging_step == 0:
                mean_loss = running_loss / logging_step
                accuracy = running_corrects / num_total_masks
                tradeoff_score = time_to_loss_score(num_steps, mean_loss, lamb=10e5)
                print(f"\n--- training score ---")
                current_m_rate = "Masking Rate: " + str(masking_rate) if masking_cl else ""
                print("Steps", num_steps, "Loss: ", mean_loss, "Accuracy: ", accuracy.item(), current_m_rate)
                #write the result to files
                with open(os.path.join(log_dir, f"log/train_{log_name}.txt"), "a", encoding="utf-8") as f:
                    f.write(f"{num_steps}\t{mean_loss}\t{accuracy.item()}\t{training_time}\n")
                #initialization
                running_loss = 0.0
                running_corrects = 0
                num_total_masks = 0
                training_time = 0.0
                #save best model
            if num_steps % logging_step == 0:
                #evaluation
                best_eval_loss = evaluate_mlm(
                    model,
                    block_size,
                    batch_size,
                    tokenizer,
                    eval_file,
                    log_dir,
                    device,
                    log_name,
                    num_steps,
                    best_eval_loss,
                    override_checkpoint,
                    masking_rate,
                )
                #baby step
                if baby_step:
                    if not update_of_loss:
                        patience_counter+=1
                        if patience_counter == baby_step_patience:
                            return model, num_steps, best_eval_loss#end training loop
                    else:
                        patience_counter=0

            if num_steps in cl_schedule and block_size != 512:
                return model, num_steps, best_eval_loss#end training loop with the current difficulty level
            #vocab CL
            if sample_id == vocab_schedule_steps:
                print("-"*10 + " Reached the limitation of available vocaburary samples " + "-"*10)
                print("the ratio of available data: ", scheduler_function(num_steps-1))
                break
            #file.seek(0)
        #lr_scheduler
        #lr_scheduler.step()
        #with open(os.path.join(log_dir, f"log/learning_rate/lr_{log_name}.txt"), "a", encoding="utf-8") as f:
            #f.write(f"{lr_scheduler.get_lr()[0]}\n")
            #when training is over
            if num_steps >= num_total_steps:
                torch.save({
                            "step": num_steps,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss": mean_loss,
                            }, os.path.join(log_dir, f"model/last_model_{log_name}.pt"))
                print("Saved the last model...")
                return (None, None, None)
