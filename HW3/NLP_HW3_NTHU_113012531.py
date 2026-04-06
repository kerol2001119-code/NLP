#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from evaluate import load
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
import os
#  You can install and import any other libraries if needed


# In[2]:


# Some Chinese punctuations will be tokenized as [UNK], so we replace them with English ones
token_replacement = [
    ["：" , ":"],
    ["，" , ","],
    ["“" , "\""],
    ["”" , "\""],
    ["？" , "?"],
    ["……" , "..."],
    ["！" , "!"]
]


# In[3]:


tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir="./cache/")


# In[4]:


class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, trust_remote_code=True, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # Replace Chinese punctuations with English ones
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)

data_sample = SemevalDataset(split="train").data[:3]
print(f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")


# In[5]:


# Define the hyperparameters
# You can modify these values if needed
lr = 3e-5
epochs = 5
train_batch_size = 8
validation_batch_size = 8


# In[6]:


# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.
def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    
    #此處參考Gemeni(AI)寫Multi input 與 multi label打包的處理方式
    texts_a = [item['premise'] for item in batch]
    texts_b = [item['hypothesis'] for item in batch]

    inputs = tokenizer(
        texts_a,
        texts_b,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    scores = [item['relatedness_score'] for item in batch]
    judgements = [item['entailment_judgment'] for item in batch]
    
    labels_regression = torch.tensor(scores, dtype=torch.float32)
    labels_classification = torch.tensor(judgements, dtype=torch.long)
    
    labels = [labels_regression, labels_classification]

    return inputs, labels
    
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.

# TODO1-2: Define your DataLoader
dl_train = DataLoader(
    dataset=SemevalDataset(split="train"), #dataset詢問Gemeni(AI)如何套用SemevalDataset
    batch_size=train_batch_size,
    shuffle=True,           
    collate_fn=collate_fn   
)

dl_validation = DataLoader(
    dataset=SemevalDataset(split="validation"),
    batch_size=validation_batch_size,
    shuffle=False,          
    collate_fn=collate_fn
)
dl_test = DataLoader(
    dataset=SemevalDataset(split="test"),
    batch_size=validation_batch_size,
    shuffle=False,
    collate_fn=collate_fn
)


# In[16]:


# TODO2: Construct your model
class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # num_labels = kwargs.pop("num_labels", None)
        # Write your code here
        # Define what modules you will use in the model
        # Please use "google-bert/bert-base-uncased" model (https://huggingface.co/google-bert/bert-base-uncased)
        # Besides the base model, you may design additional architectures by incorporating linear layers, activation functions, or other neural components.
        # Remark: The use of any additional pretrained language models is not permitted.

        # 此處參考Gemeni(AI)Model 架構
        self.num_labels = 4

        self.bert_model_name = "google-bert/bert-base-uncased"
        self.bert = BertModel.from_pretrained(self.bert_model_name)

        # 獲取 BERT 的隱藏層大小
        self.bert_hidden_size = self.bert.config.hidden_size
        
        self.dropout = torch.nn.Dropout(0.1) 
        
        # Linear Layer
        self.dense = torch.nn.Linear(self.bert_hidden_size, self.bert_hidden_size)
        
        # Activation Function
        self.activation = torch.nn.ReLU()
        
        self.classifier = torch.nn.Linear(self.bert_hidden_size, 4)

        
    
    def forward(self, **kwargs):
        # Write your code here
        # Forward pass
        
        # 此處參考Gemeni(AI)Model forward的寫法
        # 將輸入傳遞給 BERT
        bert_output = self.bert(**kwargs)

        pooled_output = bert_output.pooler_output

        regularized_output = self.dropout(pooled_output)

        activated_output = self.dense(regularized_output)
        
        activated_output = self.activation(activated_output)

        logits = self.classifier(activated_output)

        return logits


# In[17]:


# TODO3: Define your optimizer and loss function

model = MultiLabelModel().to(device)
# TODO3-1: Define your Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# TODO3-2: Define your loss functions (you should have two)
# Write your code here
loss_fn_classification = torch.nn.CrossEntropyLoss()
loss_fn_regression = torch.nn.MSELoss()


# scoring functions
psr = load("pearsonr")
acc = load("accuracy")


# In[18]:


best_score = 0.0
for ep in range(epochs):
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop
    # Write your code here
    # 此處參考Gemeni(AI)
    save_dir = "./saved_models"
    os.makedirs(save_dir, exist_ok=True)
    
    for inputs, labels in pbar:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        regression_labels = labels[0].float().to(device)
        classification_labels = labels[1].long().to(device)

        # train your model
        # clear gradient
        optimizer.zero_grad()

        # forward pass
        model_outputs = model(**inputs) #  shape (B, 4)

        logits_regression = model_outputs[:, 0].squeeze() # shape (B,)
        logits_classification = model_outputs[:, 1:]      # shape (B, 3)

        # compute loss
        loss_r = loss_fn_regression(logits_regression, regression_labels)
        loss_c = loss_fn_classification(logits_classification, classification_labels)
        loss = loss_r + loss_c 

        # back-propagation
        loss.backward()

        # model optimization
        optimizer.step()


    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # TODO5: Write the evaluation loop
    # Write your code here
    
    all_preds_r, all_labels_r = [], []
    all_preds_c, all_labels_c = [], []

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            regression_labels = labels[0].float()
            classification_labels = labels[1].long()

            # Evaluate your model
            model_outputs = model(**inputs)

            logits_regression = model_outputs[:, 0].squeeze()
            logits_classification = model_outputs[:, 1:]

            preds_r = logits_regression.cpu()
            preds_c = torch.argmax(logits_classification, dim=1).cpu()

            all_preds_r.append(preds_r)
            all_labels_r.append(regression_labels)
            all_preds_c.append(preds_c)
            all_labels_c.append(classification_labels)

    all_preds_r = torch.cat(all_preds_r)
    all_labels_r = torch.cat(all_labels_r)
    all_preds_c = torch.cat(all_preds_c)
    all_labels_c = torch.cat(all_labels_c)
    
    # Output all the evaluation scores (PearsonCorr, Accuracy)
    pearson_corr = psr.compute(
        predictions=all_preds_r, 
        references=all_labels_r
    )['pearsonr']
    
    accuracy = acc.compute(
        predictions=all_preds_c, 
        references=all_labels_c
    )['accuracy']
    
    # print(f"F1 Score: {f1.compute()}")
    current_score = pearson_corr + accuracy
    print(f"\n--- Epoch {ep+1}/{epochs} Validation Results ---")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Pearson Correlation: {pearson_corr:.4f}")
    print(f"  Current Combined Score: {current_score:.4f}")
    print(f"  Best Combined Score So Far: {best_score:.4f}")

    # print(f"F1 Score: {f1.compute()}")
    if pearson_corr + accuracy > best_score:
        best_score = pearson_corr + accuracy
        torch.save(model.state_dict(), f'./saved_models/best_model.ckpt')


# In[ ]:


# Load the model
model = MultiLabelModel().to(device)
model.load_state_dict(torch.load(f"./saved_models/best_model.ckpt", weights_only=True))

# Test Loop
pbar = tqdm(dl_test, desc="Test")
model.eval()

# TODO6: Write the test loop
# Write your code here
# 此處參考Gemeni(AI)

all_preds_r, all_labels_r = [], []
all_preds_c, all_labels_c = [], []

with torch.no_grad(): 
    for inputs, labels in pbar:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        regression_labels = labels[0].float()
        classification_labels = labels[1].long()

        # Evaluate your model
        model_outputs = model(**inputs)

        logits_regression = model_outputs[:, 0].squeeze()
        logits_classification = model_outputs[:, 1:]

        preds_r = logits_regression.cpu()
        preds_c = torch.argmax(logits_classification, dim=1).cpu()

        all_preds_r.append(preds_r)
        all_labels_r.append(regression_labels)
        all_preds_c.append(preds_c)
        all_labels_c.append(classification_labels)

all_preds_r = torch.cat(all_preds_r)
all_labels_r = torch.cat(all_labels_r)
all_preds_c = torch.cat(all_preds_c)
all_labels_c = torch.cat(all_labels_c)

test_pearson_corr = psr.compute(
    predictions=all_preds_r, 
    references=all_labels_r
)['pearsonr']

test_accuracy = acc.compute(
    predictions=all_preds_c, 
    references=all_labels_c
)['accuracy']


print("\n" + "="*30)
print("      *** Final Test Results ***")
print(f"  Test Accuracy: {test_accuracy:.4f}")
print(f"  Test Pearson Correlation: {test_pearson_corr:.4f}")
print("="*30)


# We have loaded the best model with the highest evaluation score for you
# Please implement the test loop to evaluate the model on the test dataset
# We will have 10% of the total score for the test accuracy and pearson correlation

