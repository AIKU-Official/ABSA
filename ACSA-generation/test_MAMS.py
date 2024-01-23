from simpletransformers.seq2seq import Seq2SeqModel
import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)
import torch
import numpy as np
from tqdm import tqdm
import json

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def predict_val(model, device):
    candidate_list = ["긍정", "중립", "부정"]

    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('gogamza/kobart-base-v2')
    # with open("./MAMS/MAMS_val.txt", "r") as f:
    #     file = f.readlines()
    train_data = []
    count = 0
    total = 0
    df = pd.read_csv('../data/csv/dev.csv')
    for row in df.iterrows():
        total += 1
        # score_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        _, (x, entity, aspect, golden_polarity) = row

        input_ids = tokenizer([x] * len(candidate_list), return_tensors='pt')['input_ids']
        target_list = [f"{entity}의 {aspect}에 대한 감성은 {candi}입니다." for candi in candidate_list]

        output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()
        for i in range(3):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list2.append(score)
        score_list = score_list2
        predict = candidate_list[np.argmax(score_list)]
        if predict == golden_polarity:
            count += 1
            # print(predict, golden_polarity, count/total, count, total)

    return count/total

def predict_test(model, device):
    candidate_list = ["긍정", "중립", "부정", "없음"]
    idlabel = {i: label for i, label in enumerate(candidate_list)}
    label2id = {label: i for i, label in enumerate(candidate_list)}
    entity_property_pair = [
        '본품#품질', '제품 전체#일반', '제품 전체#품질', '본품#일반', '제품 전체#디자인',
        '본품#편의성', '제품 전체#편의성', '제품 전체#인지도', '패키지/구성품#디자인', '브랜드#일반',
        '제품 전체#가격', '패키지/구성품#편의성', '패키지/구성품#일반', '본품#다양성', '본품#디자인',
        '브랜드#품질', '패키지/구성품#품질', '브랜드#인지도', '브랜드#가격', '패키지/구성품#다양성',
        '제품 전체#다양성', '본품#가격', '브랜드#디자인', '패키지/구성품#가격', '본품#인지도'
    ]   # data label 분포도 순서

    # model = BartForConditionalGeneration.from_pretrained('./outputs/checkpoint-513-epoch-19')
    model.eval()
    model.config.use_cache = False
    tokenizer = BartTokenizer.from_pretrained('gogamza/kobart-base-v2')
    # print(tokenizer("천연치약에 대한 니즈가 있으나 기존 치약의 풍부한 거품과 화한맛, 강한 자극에 익숙하여, 천연치약 사용에 불편함을 가지고 있는 나같은 어른들을 위한 치약♡", return_tensors='pt')['input_ids'].shape)
    train_data = []
    count = 0
    total = 0

    y_true = []
    y_pred = []
    df = pd.read_csv('../data/csv/dev.csv')
    out_json = []
    for row in tqdm(df.iterrows()):
        total += 1
        # score_list = []
        score_list1 = []
        score_list2 = []
        score_list3 = []
        score_list4 = []
        score_list5 = []
        
        _, x = row
        x = x[0]
        annot_list = []
        for entity_property in entity_property_pair:
            score_list2 = []
            entity, aspect = entity_property.split('#')

            input_ids = tokenizer([x] * len(candidate_list), return_tensors='pt')['input_ids']
            target_list = [f"{entity}의 {aspect}에 대한 감성은 {candi}입니다." for candi in candidate_list]
            
            output_ids = tokenizer(target_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
            with torch.no_grad():
                output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0] # (seq_len, hidden_dim)
                logits = output.softmax(dim=-1).to('cpu').numpy()
            
            for i in range(len(candidate_list)):
                score = 0
                for j in range(logits[i].shape[0] - 2):
                    score += np.log(logits[i][j][output_ids[i][j + 1]])
                score_list2.append(score)
            score_list = score_list2

            # print(len(score_list))
            predict = candidate_list[np.argmax(score_list)]

            if predict != "없음":
                annot_list.append({f"{entity_property}": predict})
            # y_pred.append(label2id[predict])
            # y_true.append(label2id[golden_polarity])
            # if predict == golden_polarity:
            #     count += 1
                # print(predict, golden_polarity, count/total, count, total)
        
        out_json.append({
            "sentence_form": x,
            "annotation": annot_list
        })

    with open("./test_output.json", "w") as f:
        json.dump(out_json, f, indent=4, ensure_ascii=False)
    return None
    # acc = accuracy_score(y_true, y_pred)
    # precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='macro') 
    # return acc, precision, recall, f1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model2 = BartForConditionalGeneration.from_pretrained('./outputs_tmp/').to(device)
    acc, precision, recall, f1 = predict_test(model2, device)

    print("acc: ", acc)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)