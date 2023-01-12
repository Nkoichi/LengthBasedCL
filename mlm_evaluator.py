#My custom Trainer class
import torch
from tqdm import tqdm
import datetime
import os
from utils import mask_token, time_to_loss_score
from torch.utils.data import Dataset, DataLoader
from transformers import TextDataset




def evaluate_mlm(
    model,
    block_size,
    batch_size,
    tokenizer,
    eval_file,
    log_dir,
    device,
    start,
    num_steps,
    best_eval_loss,
    override_checkpoint,
    masking_rate=0.15,
    ):


    #eval dataset
    eval_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=block_size,
    )
    #dataloader
    dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    #initialization
    eval_steps = 0
    running_loss = 0.0
    running_corrects = 0
    num_total_masks = 0

    model.eval()

    #loop (data sampling)
    for inputs in dataloader:
        eval_steps += 1

        #randomly mask tokens and get labels
        inputs, labels, masked_positions = mask_token(inputs, tokenizer, masking_rate)

        #move all data to CPU or GPU memory
        inputs = inputs.to(device)
        labels = labels.to(device)
        masked_positions = masked_positions.to(device)

        #calculate loss
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            running_loss += loss.item()

        masked_logits = outputs.logits.index_select(1, masked_positions) #outputs => batch, len(masked_positions), vocab
        pred_ids = torch.max(masked_logits, 2).indices #batch, len(masked_positions)
        target_ids = labels.index_select(1, masked_positions)

        running_corrects += torch.sum(pred_ids==target_ids)
        num_total_masks += labels.size(0) * labels.size(1) #batch * number of masks
    #show loss and mlm accuracy
    mean_loss = running_loss / eval_steps
    accuracy = running_corrects / num_total_masks
    tradeoff_score = time_to_loss_score(num_steps, mean_loss, lamb=10e5)
    print(f"--- evaluation score ---")
    print("Steps", num_steps, "Loss: ", mean_loss, "Accuracy: ", accuracy.item(), "\n")
    #initialization
    running_loss = 0.0
    running_corrects = 0
    num_total_masks = 0
    #write the result to files
    with open(os.path.join(log_dir, f"log/eval_{start}.txt"), "a", encoding="utf-8") as f:
        f.write(f"{num_steps}\t{mean_loss}\t{accuracy.item()}\t{tradeoff_score}\n")
    #save best model
    if not override_checkpoint and num_steps % 50000 == 0:
        torch.save(model.state_dict(), r"baseline/model/%s_model_%d.pt" % (start, num_steps))
    if mean_loss < best_eval_loss:
        torch.save(model.state_dict(), r"baseline/model/%s_model.pt" % start)
        with open(r"baseline/log/best_loss/%s_best_loss.txt" % start, "a", encoding="utf-8") as f:
            f.write(f"{num_steps}\t{mean_loss}\n")
        best_eval_loss = mean_loss
        update_of_loss = True
    else:
        update_of_loss = False

    return best_eval_loss#, update_of_loss
