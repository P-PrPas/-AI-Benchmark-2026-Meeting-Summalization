import pandas as pd
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from pythainlp.tokenize import word_tokenize
from rouge_score import rouge_scorer
from rouge_score.tokenizers import Tokenizer

BASE_DIR = Path(__file__).resolve().parents[2]
device = "cuda" if torch.cuda.is_available() else "cpu"

def tokenize_thai(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""
    # Standard Thai word segmentation
    tokens = word_tokenize(text, engine="newmm", keep_whitespace=False)
    # print(tokens)
    return " ".join(tokens)

class ThaiSpaceTokenizer(Tokenizer):
    def tokenize(self, text):
        return text.split(" ")

def load_csv(file_path):
    df = pd.read_csv(file_path)
    
    def parse_para(x):
        if pd.isna(x) or str(x).strip() == "":
            return []
        try:
            # แยกด้วย comma และแปลงเป็น int
            return [i.strip() for i in str(x).split(",")]
        except ValueError:
            # กรณีที่มีค่าที่ไม่ใช่ตัวเลขปนมา
            return []

    if 'refs' in df.columns:
        df['refs'] = df['refs'].apply(parse_para)
        
    return df

def calculate_iou(list_pred, list_sol):
    set_pred = set(list_pred) if isinstance(list_pred, list) else set()
    set_sol = set(list_sol) if isinstance(list_sol, list) else set()
    if not set_sol: return 0.0
    return len(set_pred.intersection(set_sol)) / len(set_pred.union(set_sol))

def run_evaluation(sol: pd.DataFrame, pred: pd.DataFrame, merge='ID'):
    # 1. แปลง Input เป็น DataFrames และ Merge กันด้วย 'id'
    if len(sol) != len(pred):
        raise ValueError("จำนวนแถวของ sol และ pred ไม่เท่ากัน")
    
    df = pd.merge(sol, pred, on=merge, suffixes=('_sol', '_pred'))

    df['IoU'] = df.apply(lambda x: calculate_iou(x['refs_pred'], x['refs_sol']), axis=1)

    scorer = rouge_scorer.RougeScorer(['rougeL'], 
                                     use_stemmer=False, 
                                     tokenizer=ThaiSpaceTokenizer())

    # calculate RougeL
    sol_toks = df[f'abstractive_sol'].apply(tokenize_thai)
    pred_toks = df[f'abstractive_pred'].apply(tokenize_thai)
        
    results = [scorer.score(g, p) for g, p in zip(sol_toks, pred_toks)]
        
    df[f'rougeL'] = [r['rougeL'].fmeasure for r in results]



    # calculate SS-score
    model = SentenceTransformer(str(BASE_DIR / "weight" / "Qwen3-Embedding-8B"))
    texts = df[f'abstractive_sol'].tolist() + df[f'abstractive_pred'].tolist()
    
    embeddings = model.encode(texts,batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True)
    
    ref_emb = embeddings[0:len(texts)//2]
    pred_emb = embeddings[len(texts)//2:]
    
    scores = F.cosine_similarity(
        pred_emb,
        ref_emb, dim=1
    )
    
    df[f'SS-score'] = scores.cpu().numpy()

    # สรุปผลค่าเฉลี่ย
    metric_cols = [
        "rougeL", "SS-score",
        "IoU"
    ]
    final_report = df[metric_cols].mean().to_dict()
    
    return final_report # คืนค่าทั้ง report สรุป และ df ตัวเต็มเผื่อใช้ดูรายแถว

def calculate_final_score(metrics_dict):
    wss, wrl, wj = 0.45, 0.35, 0.2
    ss = metrics_dict['SS-score']
    rl = metrics_dict['rougeL']
    j = metrics_dict['IoU']
    return wss*ss + wrl*rl + wj*j

if __name__ == "__main__":
    sol = load_csv(BASE_DIR / "data" / "eval_sample" / "submission.csv")
    pred = load_csv(BASE_DIR / "data" / "eval_sample" / "submission.csv")
    
    matrix = run_evaluation(sol, pred)
    matrix['score'] = calculate_final_score(matrix)
    print(matrix)
