from transformers import GPT2Tokenizer, TFGPT2ForSequenceClassification
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam


def tokenize_sentences(sentences, tokenizer):
    input_ids = []
    attention_masks = []

    for s in sentences:
        inputs = tokenizer.encode_plus(s, add_special_tokens=True, max_length=64, pad_to_max_length=True,
                                       return_attention_mask=True)
        input_ids.append(inputs['input_ids'])
        attention_masks.append(inputs['attention_mask'])

    input_ids = np.array(input_ids)
    attention_masks = np.array(attention_masks)

    return input_ids, attention_masks


def process_predictions(predictions):
    prediction_answers = []
    pred_length = predictions.shape[0]
    idx = 0

    for i in range(0, pred_length - 2, 3):
        if predictions[i] == 1:
            prediction_answers.append([idx, 'A'])
        elif predictions[i + 1] == 1:
            prediction_answers.append([idx, 'B'])
        elif predictions[i + 2] == 1:
            prediction_answers.append([idx, 'C'])
        else:  # some predicitons doesn't have answers in A, B or C, all empty answers are assumed as B
            prediction_answers.append([idx, 'B'])
        idx += 1
    return prediction_answers


def main():
    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type="GPU")
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = TFGPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

    json_path = './Data/'
    train = pd.read_json(json_path + 'train.json', lines=True)
    test = pd.read_json(json_path + 'test.json', lines=True)

    train_sentences = train['text'].values
    train_labels = train['klass'].values
    test_sentences = test['text'].values

    train_input_ids, train_attention_masks = tokenize_sentences(train_sentences, tokenizer)
    test_input_ids, test_attention_masks = tokenize_sentences(test_sentences, tokenizer)

    loss = SparseCategoricalCrossentropy(from_logits=True)
    metric = SparseCategoricalAccuracy('accuracy')
    optimizer = Adam(learning_rate=2e-5, epsilon=1e-08)

    model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

    history = model.fit([train_input_ids, train_attention_masks], train_labels, batch_size=32, epochs=4)

    model.save_weights('./Model/gpt2/gpt2_weights')

    # model.load_weights('./Model/gpt2/gpt2_weights')

    preds = model.predict([test_input_ids, test_attention_masks])
    pred_labels = preds[0].argmax(axis=1)
    # print(type(bert_pred_labels))
    # print(bert_pred_labels.shape)
    print("Done predictions")
    prediction_answers = process_predictions(pred_labels)
    subtaskB_pred_answers = pd.DataFrame(prediction_answers)
    subtaskB_pred_answers.columns = ['id', 'ans']
    subtaskB_pred_answers.to_csv('./Data/subtaskB_pred_gpt2_answers.csv', index=False)


if __name__ == '__main__':
    main()
