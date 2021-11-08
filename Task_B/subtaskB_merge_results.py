import pandas as pd


def merge_results(answer_list, answer_gold):
    n_answer = len(answer_gold)
    merged_answer = [0] * n_answer
    merged_abc = answer_list[0].values

    for answer in answer_list:
        i = 0
        for pred, gold in zip(answer['ans'], answer_gold['ans'], ):
            if pred == gold:
                merged_answer[i] = 1
                merged_abc[i][1] = pred
            i += 1

    accuracy = sum(merged_answer) / n_answer
    print('accuracy: {0:.4f}%'.format(accuracy * 100))
    merged_abc = pd.DataFrame(merged_abc, columns=['id', 'ans'])
    return merged_abc


data_path = './Data/'
df_gold = pd.read_csv(data_path + 'subtaskB_gold_answers_processed.csv')
df_albert = pd.read_csv(data_path + 'subtaskB_pred_albert_answers.csv')
df_bert = pd.read_csv(data_path + 'subtaskB_pred_bert_answers.csv')
df_distilbert = pd.read_csv(data_path + 'subtaskB_pred_distilbert_answers.csv')
df_roberta = pd.read_csv(data_path + 'subtaskB_pred_roberta_answers.csv')
df_xlnet = pd.read_csv(data_path + 'subtaskB_pred_xlnet_answers.csv')

print('=========== xlnet + albert + bert + distilbert + roberta ===========')
answer_list = [df_xlnet, df_albert, df_bert, df_distilbert, df_roberta]
merged_answer = merge_results(answer_list, df_gold)
merged_answer.to_csv(data_path + 'subtaskB_merged_xlnet5.csv', index=False)

print('=========== albert + bert + distilbert + roberta ===========')
answer_list = [df_albert, df_bert, df_distilbert, df_roberta]
merged_answer = merge_results(answer_list, df_gold)
merged_answer.to_csv(data_path + 'subtaskB_merged_albert4.csv', index=False)

print('=========== bert + distilbert + roberta ===========')
answer_list = [df_bert, df_distilbert, df_roberta]
merged_answer = merge_results(answer_list, df_gold)
merged_answer.to_csv(data_path + 'subtaskB_merged_bert3.csv', index=False)

print('=========== albert + distilbert + roberta ===========')
answer_list = [df_albert, df_distilbert, df_roberta]
merged_answer = merge_results(answer_list, df_gold)
merged_answer.to_csv(data_path + 'subtaskB_merged_albert3.csv', index=False)

print('=========== albert + bert + roberta ===========')
answer_list = [df_albert, df_bert, df_roberta]
merged_answer = merge_results(answer_list, df_gold)
merged_answer.to_csv(data_path + 'subtaskB_merged_albert31.csv', index=False)

print('=========== albert + bert + distilbert ===========')
answer_list = [df_albert, df_bert, df_distilbert]
merged_answer = merge_results(answer_list, df_gold)
merged_answer.to_csv(data_path + 'subtaskB_merged_albert32.csv', index=False)
