import json
import torch
import os
from sklearn.preprocessing import OneHotEncoder
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score
from transformers import Trainer

from transformers import LongformerTokenizerFast
from transformers import Trainer, TrainingArguments, LongformerForSequenceClassification
from transformers import EarlyStoppingCallback
from transformers import AutoConfig

import argparse

# os.environ["OMP_NUM_THREADS"] = '8' # 사용할 thread 만큼 부여


answer_dict = {
    'Normal Cognition': 'C',
    'Mild Cognitive Impairment': 'B',
    'Dementia': 'A',
    'Alzheimer\'s Disease' :'A'
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--eval_path", type=str)
    parser.add_argument("--input", type=str, help="patient_description / w_rationales ") # patient_description : P만 input으로 사용한 경우, w_rationales : (P, D)를 input으로 사용한 경우
    
    args = parser.parse_args()
    return args

def load_json(data_path:str):
    with open(data_path, "r", encoding="UTF-8") as f:
        data = json.load(f)
    return data

def list_up(data, input:str):
    p_list = [patient["patient_data"] for patient in data]
    r_list = []
    if input == "ADNI_patient_description":
        l_list = [patient["label"] for patient in data]
    else:  
        r_list = [patient["rationale"] for patient in data]
        l_list = [patient["org_label"] for patient in data] 
        # for d in data:
            # r_list.append( [val['output'] for key, val in d.items() if 'output_' in key] )
    return p_list, r_list, l_list  
        
def gold_label_encoding(l_list):
    labels = [[label] for label in l_list]

    encoder = OneHotEncoder(sparse=False)
    encoded_labels = encoder.fit_transform(labels) # One-Hot Encoder 생성 및 적합
    return encoded_labels.tolist()

def decode_labels(encoded_labels, labels = ['Dementia', 'Mild Cognitive Impairment', 'Normal Cognition', 'Alzheimer\'s Disease']):
    # eval_pred에서 각 환자별 라벨이 각각 어떻게 나왔는지 확인하기 위해 작성
    decoded = [labels[np.argmax(row)] for row in encoded_labels]
    return decoded

def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    res = {}
    cols = ['Dementia', 'Mild Cognitive Impairment', 'Normal Cognition']
    total_accuracy = (logits.argmax(axis=1) == labels.argmax(axis=1)).mean()
    res['Total Accuracy'] = total_accuracy
    for i, d in enumerate(cols): # num_labels : 3 (AD / MCI / CN)
        res[f"{d}_AUROC"] = roc_auc_score(labels[:, i], sigmoid(logits[:, i]))
    return res


class ReadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
    
class MulticlassTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels)) 
        return (loss, outputs) if return_outputs else loss

    
def train(args):

    # load the data
    train_path = args.train_path
    train_data = load_json(train_path)
    test_path = args.eval_path
    test_data = load_json(test_path)

    p_list_train, r_list_train, l_list_train = list_up(train_data, args.input)
    p_list_test, r_list_test, l_list_test =list_up(test_data, args.input)

    if args.input == "student_patient_description":
        train_input = p_list_train
        test_input = p_list_test
        
    elif args.input == "ADNI_patient_description":
        train_input = p_list_train
        test_input = p_list_test


    elif args.input == "w_rationales":
        # train_input = [ p_test + "".join(r_list_train[i])    for i, p_test in enumerate(p_list_test) ]

        combined_list = []
        for i in range(len(p_list_train)):
            combined_list.append(p_list_train[i]+"".join(r_list_train[i]))
        train_input = combined_list

        combined_list = []
        for i in range(len(p_list_test)):
            combined_list.append(p_list_test[i]+"".join(r_list_test[i]))
        test_input = combined_list

    else:
        print("\n\nWrong \"input\" argument\n\n")
    

    
    # load Clinical-Longformer
    ltokenizer = LongformerTokenizerFast.from_pretrained('yikuan8/Clinical-Longformer')
    model = LongformerForSequenceClassification.from_pretrained('yikuan8/Clinical-Longformer',num_labels = 3)
    
    train_encodings = ltokenizer(train_input, truncation=True, padding="max_length", max_length=4096)
    test_encodings = ltokenizer(test_input, truncation=True, padding="max_length", max_length=4096)
    
    train_dataset = ReadDataset(train_encodings, gold_label_encoding(l_list_train))
    test_dataset = ReadDataset(test_encodings, gold_label_encoding(l_list_test))
    
    training_args = TrainingArguments(
        output_dir = os.path.join(args.output_dir, args.input, "checkpoints"),
        logging_dir = os.path.join(args.output_dir, args.input, "log"),
        num_train_epochs=10,              # total number of training epochs
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        # per_device_train_batch_size=train_batch_size,  # batch size per device during training
        # per_device_eval_batch_size=eval_batch_size,   # batch size for evaluation
        evaluation_strategy='steps',      # Evaluation strategy to adopt during training
        eval_steps=20,                   # Evaluation step
        save_strategy="steps",            # Save strategy
        save_steps=20,                   # Save step
        load_best_model_at_end=True,
        fp16=True,
        fp16_backend="amp"    
    )
    
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0)

    trainer = MulticlassTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,
        callbacks=[early_stopping_callback],
        compute_metrics=compute_metrics,
    )
    
    
    model_name = "/home/taeyoon/nas2/Medical-intern/Clinical-Longformer-final/fine-tuned/w_rationales/model"  # 현재 사용 중인 모델 이름
    model = LongformerForSequenceClassification.from_pretrained(model_name, num_labels=3)
    
    # Evaluate the model
    eval_pred, outputs = trainer.evaluate(return_outputs=True)

    # Outputs를 eval_pred에 추가
    eval_pred['outputs'] = outputs
    print(outputs)

    # 나머지 코드는 이전과 동일합니다.
    eval_pred['classification_result'] = decode_labels(eval_pred['predictions'])
    eval_pred['input_data'] = test_input
    eval_result_save_path = os.path.join(args.output_dir, "fine-tuned_0131", args.input, "eval_pred.json")

    if not os.path.exists(eval_result_save_path):
         dir = os.path.join(args.output_dir, "fine-tuned_0131", args.input)
         os.makedirs(dir)
        
    with open(eval_result_save_path, "w") as file:
        json.dump(eval_pred, file, indent=4)
    print(f"==================== prediction result saved in {eval_result_save_path} ====================")

    ###### Save the fine-tuned model ######
    # trainer.save_model(os.path.join(args.output_dir, "fine-tuned", args.input, "model"))
    # print(f"==================== model saved in {os.path.join(args.output_dir, 'fine-tuned_0131', args.input, 'model')} ====================")

if __name__ == "__main__":
    args = parse_args()
    train(args)