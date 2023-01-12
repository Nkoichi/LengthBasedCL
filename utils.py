#utility functions 
import torch
import random
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import json
from pathlib import Path

#randomly mask tokens
def mask_token(inputs, tokenizer, masking_rate=0.15):
    num_of_masks = round(inputs.size(1) * masking_rate)
    if num_of_masks < 1:
        num_of_masks = 1 #mask one token at least
    #create mask matrix (masked_positions)
    masked_positions = torch.LongTensor(random.sample(range(1, inputs.size(1)), num_of_masks))
    mask_value = tokenizer.convert_tokens_to_ids("<mask>")
    #original inputs => labels
    labels = copy.deepcopy(inputs)
    return inputs.index_fill_(1, masked_positions, mask_value), labels, masked_positions

#masking rate CL
def masking_rate_linear(num_steps, total_steps, start_rate=0.05, max_rate=0.15, decrease=False):
      if not decrease:
          m_rate = start_rate + (max_rate - start_rate) * (num_steps / total_steps)
          m_rate = min(m_rate, max_rate)
      else:
          m_rate = max_rate - (max_rate - start_rate) * (num_steps / total_steps)
      return m_rate


#calculate the score of trade off between training time and mlm accuracy
def time_to_loss_score(training_steps, loss, lamb):
    return (10-loss) / (training_steps + lamb)


#visualize each metrics
def visualize_metrics(log_file, phase="training"):
    with open(log_file) as f:
        data = f.read().strip().splitlines()
    steps = [float(x.split("\t")[0]) for x in data]
    losses = [float(y.split("\t")[1]) for y in data]
    acc = [float(y.split("\t")[2]) for y in data]
    #tradeoff = [float(y.split("\t")[3]) for y in data]
    fig1 = plt.figure()
    fig2 = plt.figure()
    #fig3 = plt.figure()
    ax1 = fig1.add_subplot(111, title=f"{phase} loss.", xlabel="number of steps", ylabel="loss")
    ax2 = fig2.add_subplot(111, title=f"{phase} acc.", xlabel="number of steps", ylabel="mlm accuracy")
    #ax3 = fig3.add_subplot(111, title=f"trade off between time and accuracy in {phase}", xlabel="number of steps", ylabel="trade off score")
    ax1.plot(steps, losses)
    ax2.plot(steps, acc)
    #ax3.plot(steps, tradeoff)
    plt.show()

#compare each metrics
def visualize_comparison(file1, file2, phase="training"):
    #read file1
    with open(file1) as f:
        data1 = f.read().strip().splitlines()
    steps1 = [float(x.split("\t")[0]) for x in data1]
    losses1 = [float(y.split("\t")[1]) for y in data1]
    acc1 = [float(y.split("\t")[2]) for y in data1]
    #read file2
    with open(file2) as f:
        data2 = f.read().strip().splitlines()
    steps2 = [float(x.split("\t")[0]) for x in data2]
    losses2 = [float(y.split("\t")[1]) for y in data2]
    acc2 = [float(y.split("\t")[2]) for y in data2]
    #create new figures
    fig_loss = plt.figure()
    fig_acc = plt.figure()
    ax_loss = fig_loss.add_subplot(111, title=f"{phase} loss.", xlabel="number of steps", ylabel="loss")
    ax_acc = fig_acc.add_subplot(111, title=f"{phase} acc.", xlabel="number of steps", ylabel="mlm accuracy")
    #plot loss
    ax_loss.plot(steps1, losses1)
    ax_loss.plot(steps2, losses2)
    #plot accuracy
    ax_acc.plot(steps1, acc1)
    ax_acc.plot(steps2, acc2)

    plt.show()


#compare each metrics (extention)
def visualize_comparison_ex(*args):
    #empty list
    steps_list = []
    losses_list = []
    acc_list = []
    name_list = []
    for file, name in args:
        #read file1
        with open(file) as f:
            data = f.read().strip().splitlines()
        steps_list.append([float(x.split("\t")[0]) for x in data])
        losses_list.append([float(y.split("\t")[1]) for y in data])
        acc_list.append([float(y.split("\t")[2]) for y in data])
        name_list.append(name)


    #create new figures
    fig_loss = plt.figure()
    #fig_acc = plt.figure()
    ax_loss = fig_loss.add_subplot(111, xlabel="number of steps", ylabel="loss")
    #ax_acc = fig_acc.add_subplot(111, title=f"accuracy", xlabel="number of steps", ylabel="mlm accuracy")
    for steps, losses, acc, name in zip(steps_list, losses_list, acc_list, name_list):
        #plot loss
        ax_loss.plot(steps, losses, label=name)
        #plot accuracy
        #ax_acc.plot(steps, acc, label=name)

    plt.legend()
    plt.show()

#compare each metrics (extention)
def visualize_comparison_group(*args):
    #empty list
    steps_list = []
    losses_list = []
    acc_list = []
    label_list = []
    color_list = []
    for per_x, color, label in args:
        for file, name in per_x:
        #read file1
            with open(file) as f:
                data = f.read().strip().splitlines()
            steps_list.append([float(x.split("\t")[0]) for x in data])
            losses_list.append([float(y.split("\t")[1]) for y in data])
            acc_list.append([float(y.split("\t")[2]) for y in data])
            label_list.append(label)
            color_list.append(color) #["r", "g", "b", "c", "m", "y", "k", "w"]


    #create new figures
    fig_loss = plt.figure()
    #fig_acc = plt.figure()
    ax_loss = fig_loss.add_subplot(111, xlabel="number of steps", ylabel="loss")
    #ax_acc = fig_acc.add_subplot(111, title=f"accuracy", xlabel="number of steps", ylabel="mlm accuracy")
    for steps, losses, acc, label, color in zip(steps_list, losses_list, acc_list, label_list, color_list):
        #plot loss
        ax_loss.plot(steps, losses, label=label, color=color)
        #plot accuracy
        #ax_acc.plot(steps, acc, label=name)

    #plt.ylim(0.4, 1.2)
    plt.legend()
    plt.show()


def training_time_vis(*args):
    name_list = []
    mean_training_time = []
    for file, name in args:
        with open(file, encoding="utf-8") as f:
            data = f.read().strip().splitlines()
        mean_training_time.append(sum([float(x.split("\t")[-1]) for x in data]) / len(data))
        name_list.append(name)

    plt.bar(name_list, mean_training_time)
    plt.show()


def preprocess_inputs(inputs, tokenizer):
    start_id = tokenizer.convert_tokens_to_ids("<s>") #0

    processed_inputs = []
    for i in range(len(inputs)):
        token_ids = [int(token_id) for token_id in inputs[i].split(" ")[:-1]] #remove the last black in input
        token_ids = [start_id] + token_ids
        processed_inputs.append(token_ids)
    return torch.LongTensor(processed_inputs) #the list of list of token ids


def get_lr_scheduler(optimizer, num_training_steps, warmup_ratio=0.1, last_epoch=-1):
    num_warmup_steps = num_training_steps * 0.1
    #function for lr_scheduler
    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return current_step / num_warmup_steps
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)



#the function for reading SQuAD 2.0
def read_squad(path):
    path = Path(path)
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    contexts = []
    questions = []
    answers = []
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)

    return contexts, questions, answers

#get the character position at which the answer ends in the passage (we are given the starting position)
def add_end_idx(answers, contexts):
    for answer, context in zip(answers, contexts):
        gold_text = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(gold_text)

        # sometimes squad answers are off by a character or two – fix this
        if context[start_idx:end_idx] == gold_text:
            answer['answer_end'] = end_idx
        elif context[start_idx-1:end_idx-1] == gold_text:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
        elif context[start_idx-2:end_idx-2] == gold_text:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

#convert character start/end positions to token start/end positions
def add_token_positions(encodings, answers, tokenizer):
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


#metric for SQuAD
def squad_metric(start_pred, end_pred, start_pos, end_pos):
    sp, ep = (start_pred, end_pred) if start_pred <= end_pred else (end_pred, start_pred)
    if sp == ep:
        ep += 1
    if start_pos == end_pos:
        end_pos += 1
    pred_span = {i for i in range(sp, ep)}
    ref_span = {k for k in range(start_pos, end_pos)}
    intersection = len(pred_span & ref_span)
    precision = intersection / len(pred_span)
    recall = intersection / len(ref_span)
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    return precision, recall, f1


#show the best GLUE scores
"""
def show_best_glue(log_file):
    open(log_file, encoding="utf-8") as f:
        data = f.read().strip().splitlines()
    task_to_score = defaultdict(list)
    for line in data:
        elements = line,split("\t")
        task = elements[0]
        if not task in ["cola", "qqp", "stsb"]:
            score = elements[1] #accuracy
        elif task == "cola":
            score = elements[5] #MCC
        elif task == "qqp"
            score = (elements[4], elements[1])
        elif task == "stsb": #pearson & spearmann
            score = (elements[1], elements[2])

        task_to_score[task].append(score))
        #max
"""