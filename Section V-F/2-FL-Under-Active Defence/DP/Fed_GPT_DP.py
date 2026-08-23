import torch
from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_dataset
import time
import numpy as np
import random
import copy
from auxiliary_function import utility_function, GPT_insert_Adapter


def split_dataset_evenly(dataset, num_splits):
    length = len(dataset)
    lengths = [length // num_splits] * num_splits
    for i in range(length % num_splits):
        lengths[i] += 1
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))


class SSTDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        sentence = item['sentence']
        label = item['label']

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_count = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            correct_predictions += (preds == labels).sum().item()
            total_count += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_count
    return avg_loss, accuracy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = "E:/Model/GPT2"
max_length = 128
batch_size = 1
learning_rate = 2e-5
num_clients = 30
num_rounds = 4800
test_every_rounds = 80
target_train_size = 70000
per_s = [1.5]

train_loss_list = []
accuracy_list = []
time_list = []
round_time_list = []
start_training_time = time.time()

print("正在加载 CoLA 数据集...")
dataset = load_dataset('glue', 'cola')

raw_train_dataset = dataset['train']
raw_test_dataset = dataset['validation']

total_train_samples = len(raw_train_dataset)
if total_train_samples > target_train_size:
    indices = list(range(total_train_samples))
    random.shuffle(indices)
    train_subset_indices = indices[:target_train_size]
    train_dataset = Subset(raw_train_dataset, train_subset_indices)
else:
    train_dataset = raw_train_dataset

test_dataset = raw_test_dataset

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(raw_test_dataset)}")
client_datasets = split_dataset_evenly(train_dataset, num_clients)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

client_dataloaders = []
for client_data in client_datasets:
    processed_data = SSTDataset(client_data, tokenizer, max_length)
    dataloader = DataLoader(processed_data, batch_size=batch_size, shuffle=True)
    client_dataloaders.append(dataloader)

test_dataset = SSTDataset(raw_test_dataset, tokenizer, max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for std in per_s:
    print("正在初始化 GPT 模型...")
    global_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    global_model.config.pad_token_id = tokenizer.pad_token_id
    GPT_insert_Adapter.insert_transformer_adapters(global_model, reduction_factor=2, layer_start=0, layer_end=len(global_model.transformer.h))
    GPT_insert_Adapter.set_trainable_parameters(global_model)
    global_model.to(device)

    print(f"开始联邦学习训练（基于梯度聚合），共 {num_rounds} 轮...")
    global_optimizer = AdamW(global_model.parameters(), lr=learning_rate, eps=1e-8)

    for round_idx in range(num_rounds):
        print(f"\n--- 联邦通信轮次 [{round_idx + 1} / {num_rounds}] ---")
        round_start_time = time.time()

        client_gradients = []
        client_batch_losses = []

        selected_clients = list(range(num_clients))

        for client_id in selected_clients:
            local_model = copy.deepcopy(global_model)
            local_model.train()
            optimizer = AdamW(local_model.parameters(), lr=learning_rate, eps=1e-8)

            dataloader_iter = iter(client_dataloaders[client_id])
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(client_dataloaders[client_id])
                batch = next(dataloader_iter)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if client_id == 0 and round_idx % 10 == 0:
                ref_training_data = batch

            outputs = local_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            client_batch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()

            param_gradients = {}
            for name, param in local_model.named_parameters():
                if param.requires_grad:
                    clean_grad = param.grad.data.clone().cpu()
                    param_gradients[name] = clean_grad
                    noisy_grad = utility_function.add_gaussian_noise(clean_grad, std=std)
                    param_gradients[name] = noisy_grad

            client_gradients.append(param_gradients)
            del local_model, optimizer
            torch.cuda.empty_cache()

        print("  正在聚合来自 30 个客户端的梯度...")
        global_model.train()
        global_optimizer.zero_grad()

        for name, param in global_model.named_parameters():
            if not param.requires_grad:
                continue
            grad_list = []
            for client_grad_dict in client_gradients:
                if name in client_grad_dict:
                    grad_list.append(client_grad_dict[name].to(device))

            if grad_list:
                avg_grad = torch.stack(grad_list, dim=0).mean(dim=0)
                param.grad = avg_grad

        global_optimizer.step()

        avg_client_loss = np.mean(client_batch_losses)
        current_time = time.time() - start_training_time
        train_loss_list.append(avg_client_loss)
        time_list.append(current_time)
        round_time_list.append(time.time() - round_start_time)
        print(f"  本轮平均客户端 loss: {avg_client_loss:.4f}")

        if (round_idx + 1) % test_every_rounds == 0 or round_idx == num_rounds - 1:
            print(f"  正在测试全局模型性能...")
            test_loss, test_acc = evaluate_model(global_model, test_dataloader, device)
            accuracy_list.append((round_idx + 1, test_loss, test_acc))
            print(f"  测试完成 | Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")

    np.save(f'std_{std}_fed_sgd_bert_train_loss.npy', np.array(train_loss_list))
    np.save(f'std_{std}_fed_sgd_bert_time_sec.npy', np.array(time_list))
    np.save(f'std_{std}_fed_sgd_bert_round_times.npy', np.array(round_time_list))
    np.save(f'std_{std}_fed_sgd_bert_accuracy_log.npy', np.array(accuracy_list, dtype=object))

    print("训练完成！所有数据已保存。")
