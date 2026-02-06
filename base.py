import os
from tqdm import tqdm
import pandas as pd
import re
from rank_bm25 import BM25Okapi
from itertools import combinations
import torch
from transformers import AutoTokenizer
from glob import glob

base_path = './open'
train_path = os.path.join(base_path, 'train_code')
data_path = os.path.join(base_path, 'dataframe')
# 전처리
import re

def data_clean(text):
    # 중복 줄 바꿈 제거
    

    # 주석 제거
    text = re.sub(r'//.*', '', text)  # 한 줄 주석 제거
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)  # 여러 줄 주석 제거
    
    # #include 
    text = re.sub(r'#include.*', '', text)

    text = re.sub(r'<vector>', '', text)
    text = re.sub(r'vector<.*?>', '', text)
    text = re.sub(r'typedef.*;', '', text)
    text = re.sub(r'template<.*?>', '', text)
    
    # using namespace std 제거
    text = re.sub(r'using namespace std;', '', text)
    
    # 데이터 타입 형식 제거 (int, long, char 등)
    text = re.sub(r'\b(int|long|char|short|float|double|bool|void|unsigned|signed)\b', '', text)
    
    # 특수 문자 제거 (#)
    
    # const 제거
    text = re.sub(r'\bconst\b', '', text)
    
    # pragma 제거
    text = re.sub(r'#pragma.*', '', text)
    
    text = re.sub(r'\n+', '\n', text)

    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\t', ' ')
    
    # 빈 줄 제거
    text = text.strip()
    
    return text

exam = '''
#include<iostream>
<vector>
using namespace std;
typedef long long li;
#define repa(i,a,n) for(int i=(a);i<(n);i++)
#define rep(i,n) for(int i=0;i<(n);i++)
#define df 0
template<class T> void print(const T& t){ cout << t << "\n"; }
template<class T, class... Ts> void print(const T& t, const Ts&... ts) { cout << t; if (sizeof...(ts)) cout << " "; print(ts...); }

int main(){
  int n; cin >>n;
  vector<int> c(n-1),s(n-1),f(n-1);
  rep(i,n-1){
    cin >>c[i] >>s[i] >>f[i];
  }
  rep(j,n){
    int t=0;
    repa(i,j,n-1){
      //      if(df)print(s[i],f[i]);
      if(t<s[i])t=s[i];
      if(t%f[i])t=(t/f[i]+1)*f[i];
      t+=c[i];
    }
    print(t);
  }
}
'''

cleaned_exam = data_clean(exam)
print(cleaned_exam)

# p_df 생성
import pandas as pd
import os
from tqdm import tqdm
import glob

code_list = []
p_num_list = []

# 파일들의 경로를 가져오기 위해 glob 사용
for p_num, problem in enumerate(tqdm(dir_list), start=1):
    for sol in glob.glob(os.path.join(problem, '*')):
        with open(sol, 'r', encoding='utf-8') as f:
            code = f.read()
            code_list.append(data_clean(code))
            p_num_list.append(p_num)

p_df = pd.DataFrame(data={"code": code_list, "p_num": p_num_list})

# CSV 파일로 저장
p_df.to_csv(os.path.join(data_path, "problem_df_2403290000.csv"), index=False)

pd.set_option('display.max_colwidth', None)
#불러올때
p_df = pd.read_csv(os.path.join(data_path, "problem_df_2403280000.csv"))
p_df.head()
from itertools import combinations
import random

def get_pair(inputs, tokenizer):
  codes = inputs['code'].to_list()
  problems = inputs['p_num'].unique().tolist()
  problems.sort()

  tokenized_corpus = [tokenizer.tokenize(code) for code in codes]
  bm25 = BM25Okapi(tokenized_corpus)

  total_positive_pairs = []
  total_negative_pairs = []

  for problem in tqdm(problems):
    solution_codes = inputs[inputs['p_num'] == problem]['code']
    #이건 렘이 부족해서 코랩에서는 불가능한 경우 같아서 일부만 추출 해봄
    positive_pairs = list(combinations(solution_codes.to_list(), 2))
    positive_pairs = random.sample(positive_pairs, len(positive_pairs) // 20)

    solution_codes_indices = solution_codes.index.to_list()
    negative_pairs = []

    first_tokenized_code = tokenizer.tokenize(positive_pairs[0][0])
    negative_code_scores = bm25.get_scores(first_tokenized_code)
    negative_code_ranking = negative_code_scores.argsort()[::-1]
    ranking_idx = 0

    for solution_code in solution_codes:
      negative_solutions = []
      while len(negative_solutions) < len(positive_pairs) // len(solution_codes):
        high_score_idx = negative_code_ranking[ranking_idx]

        if high_score_idx not in solution_codes_indices:
          negative_solutions.append(inputs['code'].iloc[high_score_idx])
        ranking_idx += 1

      for negative_solution in negative_solutions:
        negative_pairs.append((solution_code, negative_solution))

    total_positive_pairs.extend(positive_pairs)
    total_negative_pairs.extend(negative_pairs)

  positive_code1 = list(map(lambda x:x[0], total_positive_pairs))
  positive_code2 = list(map(lambda x:x[1], total_positive_pairs))

  negative_code1 = list(map(lambda x:x[0], total_negative_pairs))
  negative_code2 = list(map(lambda x:x[1], total_negative_pairs))

  positive_label = [1] * len(positive_code1)
  negative_label = [0] * len(negative_code1)

  positive_code1.extend(negative_code1)
  positive_code2.extend(negative_code2)
  positive_label.extend(negative_label)

  pair_data = pd.DataFrame(data = {
      'code1' : positive_code1,
      'code2' : positive_code2,
      'similar' : positive_label
  })

  pair_data = pair_data.sample(frac=1).reset_index(drop=True)

  return pair_data


from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    p_df,
    test_size = 0.05,
    random_state = 42,
    stratify = p_df['p_num']
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

train_df
# bm_25 train,val 생성
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('neulab/codebert-cpp')
tokenizer.truncation_side = 'left'

# train_df와 val_df 데이터프레임 복사
train_df_copy = train_df.copy()
val_df_copy = val_df.copy()

# 토크나이저를 사용하여 코드를 토큰화
train_df_copy['code_tokenized'] = train_df_copy['code'].apply(lambda x: tokenizer.tokenize(x))
val_df_copy['code_tokenized'] = val_df_copy['code'].apply(lambda x: tokenizer.tokenize(x))

# 코드 길이를 최대 512로 제한
train_df_copy['code_tokenized'] = train_df_copy['code_tokenized'].apply(lambda x: x[:512])
val_df_copy['code_tokenized'] = val_df_copy['code_tokenized'].apply(lambda x: x[:512])

# 새로운 데이터프레임을 사용하여 pair 생성 함수 호출
bm25_train_df = get_pair(train_df_copy, tokenizer)

# 결과 출력
bm25_train_df

def generate_pairs(inputs, tokenizer):
    total_positive_pairs = []
    total_negative_pairs = []

    for problem in tqdm(inputs['p_num'].unique(), desc="Generating pairs"):
        solution_codes = inputs[inputs['p_num'] == problem]['code']
        if len(solution_codes) < 2:
            continue  # 솔루션 코드가 2개 미만인 경우 쌍을 생성할 수 없음

        # 긍정적인 쌍 생성
        positive_pairs = list(combinations(solution_codes.tolist(), 2))
        total_positive_pairs.extend(positive_pairs)

        # 부정적인 쌍 생성
        for solution_code in solution_codes:
            other_solution_codes = solution_codes[solution_codes != solution_code]
            for other_solution_code in other_solution_codes:
                total_negative_pairs.append((solution_code, other_solution_code))

    return total_positive_pairs, total_negative_pairs



def get_pair_val(val_df, tokenizer):
    # 코드 쌍 생성
    positive_pairs, negative_pairs = generate_pairs(val_df, tokenizer)

    # 데이터프레임 생성
    if len(positive_pairs) == 0 or len(negative_pairs) == 0:
        return pd.DataFrame(columns=['code1', 'code2', 'similar'])

    pair_data = pd.DataFrame({
        'code1': [],
        'code2': [],
        'similar': []
    })

    # 데이터프레임에 데이터 추가
    pair_data['code1'] = [pair[0] for pair in positive_pairs] + [pair[0] for pair in negative_pairs]
    pair_data['code2'] = [pair[1] for pair in positive_pairs] + [pair[1] for pair in negative_pairs]
    pair_data['similar'] = [1] * len(positive_pairs) + [0] * len(negative_pairs)

    # 데이터프레임을 임의로 섞음
    pair_data = pair_data.sample(frac=1).reset_index(drop=True)

    return pair_data

bm25_val_df = get_pair_val(val_df_copy, tokenizer)
bm25_train_df.to_csv(os.path.join(data_path, "bm25_train_df_2403262033.csv"), index=False)
bm25_val_df.to_csv(os.path.join(data_path, "bm25_val_df_2403262033.csv"), index=False)
bm25_train_df
# 해당 코드는 test.df 생성코드

import pandas as pd
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('neulab/codebert-cpp')
tokenizer.truncation_side = 'left'

# test.csv 파일을 읽어들임
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

# 데이터프레임을 복사
test_df_copy = test_df.copy()

# 코드를 토큰화하여 새로운 열에 저장
test_df_copy['code1_tokenized'] = test_df_copy['code1'].apply(lambda x: tokenizer.tokenize(x)[:512])
test_df_copy['code2_tokenized'] = test_df_copy['code2'].apply(lambda x: tokenizer.tokenize(x)[:512])

# 토큰화된 결과를 CSV 파일에 저장
test_df_copy.to_csv(os.path.join(data_path, "test_df_2403280000.csv"), index=False)

test_df

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import torch
import transformers

from glob import glob
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding, AutoModel, AutoModelForSequenceClassification,DataCollatorForTokenClassification,EarlyStoppingCallback
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from datasets import load_dataset, load_metric, Dataset

from tqdm import tqdm
from tqdm import trange
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
device
data_path = './open/dataframe/'


train_df = pd.read_csv(os.path.join(data_path, "bm25_train_df_2403262033.csv"))
val_df = pd.read_csv(os.path.join(data_path, "bm25_val_df_2403262033.csv"))
train_df = train_df.rename(columns={'similar': 'label'})
val_df = val_df.rename(columns={'similar': 'label'})
train_df
train_df = train_df.sample(n = 3000000, replace = True).reset_index(drop=True)
val_df = val_df.sample(n= 1000, replace = True).reset_index(drop = True)
dataset_train = Dataset.from_pandas(train_df)
dataset_val = Dataset.from_pandas(val_df)
train_df
import random

#model_name = 'neulab/codebert-cpp'
model_name = 'microsoft/codereviewer'
wd = 0.01
batch_size = 16
lr = 2e-5
epochs = 1
task = 'binary_classification'
label_list = ['0', '1']
num_labels = 2

def seed_everything(seed:42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    tokenizer.truncation_side = 'left'
    return tokenizer(examples["code1"], examples["code2"],padding="max_length", max_length = 512, truncation=True)
tokenized_train_datasets= dataset_train.map(tokenize_function, batched=True)
tokenized_val_datasets = dataset_val.map(tokenize_function, batched=True)
args = TrainingArguments(
    output_dir = './open/codereviewer',
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate= lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    seed = 42,
    weight_decay=wd,
    load_best_model_at_end=True,
    logging_dir = './open/codereviewer'
)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = []
    for i in pred.predictions:
      preds.append(np.argmax(i, axis=1).flatten())
    print(preds)
    acc = accuracy_score(labels, preds[0])
    return {'accuracy': acc}

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_train_datasets,
    eval_dataset=tokenized_val_datasets,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
trainer.train()
base_path = './open/codereviewer'

torch.save(model, os.path.join(base_path + '/model0401.pt'))  # 전체 모델 저장
base_path = './open/codereviewer'
data_path = './open/dataframe'
model = torch.load(os.path.join(base_path, 'model0401.pt'))

tokenizer = AutoTokenizer.from_pretrained(model_name)

test_df = pd.read_csv(os.path.join(data_path,'test_df_2403262033.csv'))
test_df = test_df.drop(columns = ['pair_id', 'code1_tokenized', 'code2_tokenized'], axis = 1)
for i in range(len(test_df)):
  if i % 10000 == 0:
    print(i)
  test_df.iloc[i]['code1'] = data_clean(test_df.iloc[i]['code1'])
  test_df.iloc[i]['code2'] = data_clean(test_df.iloc[i]['code2'])
args = TrainingArguments(
    output_dir = './open',
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate= lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    seed = 42,
    weight_decay=wd,
    load_best_model_at_end=True,
    logging_dir = './open'
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = []
    for i in pred.predictions:
      preds.append(np.argmax(i, axis=1).flatten())
    print(preds)
    acc = accuracy_score(labels, preds[0])
    return {'accuracy': acc}

trainer = Trainer(
    model,
    args,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
predictions = []
for i in trange(len(test_df) // 5000):
  pred = []
  dataset_test = Dataset.from_pandas(test_df[i*5000:(i+1)*5000])
  tokenized_test_datasets = dataset_test.map(tokenize_function, batched=True)
  predictions_test = trainer.predict(tokenized_test_datasets)

  for j in predictions_test.predictions:
    pred.append(np.argmax(j, axis=1).flatten())

  predictions.extend(pred[0])

len(predictions)
sub = pd.read_csv(os.path.join('./open', 'sample_submission.csv'))
sub['similar'] = predictions

sub.to_csv(os.path.join(base_path, 'submission_0401.csv'), index = False)