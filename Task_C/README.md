# SubtaskC

Pre-request: pytorch-transformers (up-to-date)

Script Instruction:

Execute train.sh to finetune a pretrained model, model list can be found in run_lm_finetuning.py. Please modify the train.sh script directly to use any pretrained model.
Eg:

./train.sh <exp_dir> <gradient_accumulation_steps> <num_train_epochs>

Execute generate.sh to output actual generated context, context saved in "data_dir" directory, with the name "subtaskC_answer.csv"
Eg:

./generate.py <exp_dir> # <exp_dir> is the directory where your trained model saved
