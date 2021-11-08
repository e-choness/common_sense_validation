# Group_06_Final_Project: Commonsense Validation

### Note: Kindly change the file path if required!

## Introduction 
This project is based on the * [SemEval 2020 Task 4 (ComVE) Task A.](https://competitions.codalab.org/competitions/21080). Recently utilizing natural language understanding systems to common sense has received significant attention in the research area. Nevertheless, this kind of task remains pretty complicated to solve because of the ambiguity which is one of the natural properties of language and yet to achieve performance compared with human understanding of common sense.  

# Dataset
We have used dataset that given "https://github.com/wangcunxiang/SemEval2020-Task4-Commonsense-Validation-and-Explanation" here.

# Subtask A
Run the CommonsenseValidation.py file.

# Subtask B 
Perform the following steps to successfully run the file:

* TaskB-Pre-Processing.ipynb
* subtaskB_albert.py
* subtaskB_bert.py
* subtaskB_bert_large.py
* subtaskB_distilbert.py
* subtaskB_gpt.py
* subtaskB_xlnet.py
* subtaskB_roberta.py
* subtaskB_merge_results.py

# SubtaskC

Pre-request: pytorch-transformers (up-to-date)

Script Instruction:

Execute train.sh to finetune a pretrained model, model list can be found in run_lm_finetuning.py. Please modify the train.sh script directly to use any pretrained model.
Eg:

./train.sh <exp_dir> <gradient_accumulation_steps> <num_train_epochs>

Execute generate.sh to output actual generated context, context saved in "data_dir" directory, with the name "subtaskC_answer.csv"
Eg:

./generate.py <exp_dir> # <exp_dir> is the directory where your trained model saved
 
 
## References 
* https://huggingface.co/ 
