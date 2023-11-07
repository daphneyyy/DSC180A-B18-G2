#!/usr/bin/env python
# coding: utf-8

# In[17]:


# import required packages
import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re


# In[18]:


# upload the dataset and load the data.
# this dataset is the original dataset 
# and does not contain the dates and times.
file = 'Transacation_outflows_3k.pqt'
data = pd.read_parquet(file, engine='auto')


# In[19]:


# Check if a GPU is available, and use it if possible, otherwise use the CPU
# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[20]:


# load the first tenth datas in the dataset.
data[:10]


# In[21]:


# Filter the required categories and define a new dataset
# which only contains these categories.
categories_filter = ['GENERAL_MERCHANDISE', 'FOOD_AND_BEVERAGES', 'GROCERIES', 'TRAVEL', 'PETS', 'EDUCATION', 'OVERDRAFT', 'RENT', 'MORTGAGE']
data1 = data[data['category_description'].isin(categories_filter)]


# In[22]:


# Only inlcude a subset of the dataset 
# to prevent running out of memory problem.
data2 = data1[:50000]
len(data2)


# In[28]:


# Data Cleanning Process Part


## Changing memo_clean column values to all lower case first.
data2['memo_clean'] = data2['memo_clean'].str.lower()


## Use regular expressions to remove text after ".com*" 
## and keep the preceding text from ".com"
def clean_text1(text):
    # Use regular expressions to remove text after ".com*" and keep the preceding text from ".com"
    cleaned_text = re.sub(r'\.com\*.*?(?=\s|$)', '', text)
    return cleaned_text


## Removing useless pattenrs
def remove_key_phrases(text):
    phrases = [
        'pos debit - visa check card xxxx - ',
        'purchase authorized on xx/xx',
        'pos purchase',
        'purchase',
        'pos',
        'web id',
        'terminal id',
        'id'
    ]
    for phrase in phrases:
        text = re.sub(phrase, '', text)
    return text


## Removing special characters.
def remove_special_char(text):
    return re.sub(r'[^a-zA-Z0-9\s]', ' ', text)


## Removing all the repeat 'x' patterns
def remove_xs(text):
    text = re.sub(r'(xx+)\b', ' ', text)
    text = re.sub(r'\b(x)\b', ' ', text)
    text = re.sub(r'\b(xx+)([a-zA-Z])', r'xx\2', text)
    return text


## Simplify repeating pattenrs for amazon and walmart
def standardize_phrase(text):
    text = re.sub(r'\b(amazon|amzn|amz)\b', 'amazon', text)
    text = re.sub(r'\b(wal\smart|wal|wm\ssupercenter|wm\ssuperc|wm)\b', 'walmart', text)
    return text


## Removing multiple spaces
def remove_multiple_spaces(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# In[33]:


# Applying thoese cleaning functions to the subset of the dataset
# that we choose.

data2['memo_clean'] = data2['memo_clean'].apply(clean_text1)
data2['memo_clean'] = data2['memo_clean'].apply(remove_key_phrases)
data2['memo_clean'] = data2['memo_clean'].apply(remove_special_char)
data2['memo_clean'] = data2['memo_clean'].apply(remove_xs)
data2['memo_clean'] = data2['memo_clean'].apply(standardize_phrase)
data2['memo_clean'] = data2['memo_clean'].apply(remove_multiple_spaces)


# In[36]:


# Check dataset after cleanning.
data2[:10]


# In[38]:


# Check numbers of each categories.
data2['category_description'].value_counts()


# In[40]:


# Assign labels to each categories.

labels = data2.category_description.unique()

label_dict = {}
for index, label in enumerate(labels):
    label_dict[label] = index
label_dict


# In[41]:


# Creating a label column for the dataset.

data2['label'] = data2.category_description.replace(label_dict)


# In[42]:


# Check the current dataset
data2[:10]


# In[45]:


# split dataset into train, validation and test sets using stratify.
train_text, temp_text, train_labels, temp_labels = train_test_split(data2['memo_clean'], data2['label'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=data2['label'])


val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)


# In[46]:


# Load the tokenizer from bert packages

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                          do_lower_case=True)


# In[48]:


# Tokenize the text in all train, val and test datasets.
# Set the max_length to 256 for safe.

encoded_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


encoded_val = tokenizer.batch_encode_plus(
    val_text.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)

encoded_test = tokenizer.batch_encode_plus(
    test_text.tolist(), 
    add_special_tokens=True, 
    return_attention_mask=True, 
    pad_to_max_length=True, 
    max_length=256, 
    return_tensors='pt'
)


# In[49]:


# Convert the tokenized list to tensors

input_ids_train = encoded_train['input_ids']
attention_masks_train = encoded_train['attention_mask']
labels_train = torch.tensor(train_labels.tolist())

input_ids_val = encoded_val['input_ids']
attention_masks_val = encoded_val['attention_mask']
labels_val = torch.tensor(val_labels.tolist())

input_ids_test = encoded_test['input_ids']
attention_masks_test = encoded_test['attention_mask']
labels_test = torch.tensor(test_labels.tolist())


dataset_train = TensorDataset(input_ids_train, attention_masks_train, labels_train)
dataset_val = TensorDataset(input_ids_val, attention_masks_val, labels_val)
dataset_test = TensorDataset(input_ids_test, attention_masks_test, labels_test)


# In[50]:


# Load the model and push to the device which we defined at the beginning.

model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False)
model = model.to(device)


# In[51]:


# Setting the batch size to three
# Using RandomSampler to randomly sample the training set.
# Using SequentialSampler for validation set to sequentially test the data.
# Using DataLoaer to improve efficient iteration and batching the data
# during training and validation.

batch_size = 3

dataloader_train = DataLoader(dataset_train, 
                              sampler=RandomSampler(dataset_train), 
                              batch_size=batch_size)

dataloader_validation = DataLoader(dataset_val, 
                                   sampler=SequentialSampler(dataset_val), 
                                   batch_size=batch_size)

dataloader_test = DataLoader(dataset_test, 
                                   sampler=SequentialSampler(dataset_test), 
                                   batch_size=batch_size)


# In[52]:


# Define an optimizer
# Setting the epochs to be five
optimizer = AdamW(model.parameters(),
                  lr=1e-5, 
                  eps=1e-8)
                  
epochs = 5

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0,
                                            num_training_steps=len(dataloader_train)*epochs)


# In[64]:


# Define the perfomance metrics through F1_Score and Accuracy Score

label_dict_inverse = {v: k for k, v in label_dict.items()}

## Calculate the F1 score for a multi-class classification task.
## Args: preds-Predicted labels,  labels-True labels
def f1_func(preds, labels):
    p = np.argmax(preds, axis=1).flatten()
    l = labels.flatten()
    f1 = f1_score(l, p, average='weighted')
    return f1


## Calculate and print accuracy for each class
## Calculate and print overal accuracy score
## Args: preds-Predicted labels,  labels-True labels, lab_dict_inverse-Inverse label dictionary
def accuracy_per_class(preds, labels, label_dict_inverse):
    p = np.argmax(preds, axis=1).flatten()
    l = labels.flatten()

    class_accuracies = {}
    for label in np.unique(l):
        mask = l == label
        y_preds = p[mask]
        y_true = l[mask]
        class_name = label_dict_inverse[label]
        class_accuracy = accuracy_score(y_true, y_preds)
        class_accuracies[class_name] = class_accuracy

    overall_accuracy = accuracy_score(l, p)

    # Print class accuracies
    for class_name, class_accuracy in class_accuracies.items():
        print(f'Class: {class_name}\nAccuracy: {class_accuracy:.2%}\n')

    # Print overall accuracy
    print(f'Overall Accuracy: {overall_accuracy:.2%}')


# In[55]:


seed_val = 17
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# In[58]:


# Define the evaluate function

def evaluate(dataloader_val):

    model.eval()
    
    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader_val:
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        loss = outputs[0]
        logits = outputs[1]
        loss_val_total += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = inputs['labels'].cpu().numpy()
        predictions.append(logits)
        true_vals.append(label_ids)
    loss_val_avg = loss_val_total/len(dataloader_val)

    
    predictions = np.concatenate(predictions, axis=0)
    true_vals = np.concatenate(true_vals, axis=0)
            
    return loss_val_avg, predictions, true_vals


# In[59]:


# Train the model

for epoch in tqdm(range(1, epochs+1)):
    
    model.train()
    
    loss_train_total = 0

    progress_bar = tqdm(dataloader_train, desc='Epoch {:1d}'.format(epoch), leave=False, disable=False)
    for batch in progress_bar:

        model.zero_grad()
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                 }       

        outputs = model(**inputs)
        
        loss = outputs[0]
        loss_train_total += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item()/len(batch))})
    torch.save(model.state_dict(), 'saved_weights.pt')
        
    tqdm.write(f'\nEpoch {epoch}')
    
    loss_train_avg = loss_train_total/len(dataloader_train)            
    tqdm.write(f'Training loss: {loss_train_avg}')
    
    val_loss, predictions, true_vals = evaluate(dataloader_validation)
    val_f1 = f1_func(predictions, true_vals)
    tqdm.write(f'Validation loss: {val_loss}')
    tqdm.write(f'F1 Score (Weighted): {val_f1}')


# In[65]:


# Calculating the Accuracy per calss and the overall Acurracy Score
# Caculating the precision, recall, and f1-score
_, predictions, true_vals = evaluate(dataloader_test)
accuracy_per_class(predictions, true_vals, label_dict_inverse)

preds = np.argmax(predictions, axis = 1)
print(classification_report(labels_test, preds))


# In[ ]:




