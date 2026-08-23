from auxiliary_code import GPT_with_adapter
from auxiliary_code import attack_GPT_auxiliary_function
import torch
from transformers import GPT2Tokenizer, GPT2Model
import json
from torch.utils.data import DataLoader
import os
import time
import copy
import numpy as np


def hook_embedding_output(module, input, output):
    global embedding_output
    embedding_output = input[0]


def hook_attn_inputs(module, input, output):
    global attn_inputs
    attn_inputs.append(input[0].detach())


def preprocess_a(a):
    if not isinstance(a, torch.Tensor):
        raise TypeError("Input must be torch.Tensor")
    if torch.isnan(a).any() or torch.isinf(a).any():
        raise ValueError("Input matrix a contains NaN or Inf values.")
    A = a.float()
    try:
        q_full, r_full = torch.linalg.qr(A, mode='reduced')
        singular_values = torch.linalg.svdvals(r_full)
        max_sv = singular_values.max()
        rank = (singular_values > 1e-6 * max_sv).sum().item()
        if rank == 0:
            return {'rank': 0}
        q_reduced = q_full[:, :rank]
        r_reduced = r_full[:rank, :rank]
        reg = 1e-6 * torch.eye(rank, device=A.device)
        r_reduced += reg
        pinv_A = torch.linalg.pinv(A)
        return {
            'rank': rank,
            'q_reduced': q_reduced,
            'r_reduced': r_reduced,
            'pinv_A': pinv_A,
            'A': A,
        }
    except Exception as e:
        print(f"Error in preprocessing a: {str(e)}")
        return {'rank': -1}


def find_expressible_vectors3_preprocessed(b, preprocessed_data, tol_error=1e-2):
    if not isinstance(b, torch.Tensor):
        raise TypeError("Input b must be torch.Tensor")
    if torch.isnan(b).any() or torch.isinf(b).any():
        raise ValueError("Input matrix b contains NaN or Inf values.")

    n1, n2, m = b.shape
    b_flat = b.view(-1, m)
    results = []

    pre_rank = preprocessed_data.get('rank', 0)
    if pre_rank <= 0:
        return []

    q_reduced = preprocessed_data['q_reduced']
    r_reduced = preprocessed_data['r_reduced']
    pinv_A = preprocessed_data['pinv_A']
    A = preprocessed_data['A']

    batch_size_qr = 2048
    n_samples = b_flat.shape[0]

    for start in range(0, n_samples, batch_size_qr):
        end = min(start + batch_size_qr, n_samples)
        b_batch = b_flat[start:end]

        with torch.no_grad():
            try:
                q_t_b = q_reduced.T @ b_batch.T
                solution_qr = torch.linalg.solve_triangular(r_reduced, q_t_b, upper=True)
                reconstructed_qr = (q_reduced @ solution_qr).T
            except Exception as e:
                print(f"QR failed on batch {start}:{end}, error: {str(e)}")
                valid_mask_qr = torch.zeros(b_batch.shape[0], dtype=torch.bool, device=b_batch.device)
                err_qr = torch.full((b_batch.shape[0],), float('inf'), device=b_batch.device)
            else:
                err_qr = torch.linalg.norm(b_batch - reconstructed_qr, dim=1)
                valid_mask_qr = err_qr < tol_error

            try:
                coefficients_pinv = pinv_A @ b_batch.T
                reconstructed_pinv = (A @ coefficients_pinv).T
            except Exception as e:
                print(f"PINV failed on batch {start}:{end}, error: {str(e)}")
                valid_mask_pinv = torch.zeros(b_batch.shape[0], dtype=torch.bool, device=b_batch.device)
                err_pinv = torch.full((b_batch.shape[0],), float('inf'), device=b_batch.device)
            else:
                err_pinv = torch.linalg.norm(b_batch - reconstructed_pinv, dim=1)
                valid_mask_pinv = err_pinv < tol_error

        valid_mask = valid_mask_qr | valid_mask_pinv
        recon_qr_clamped = torch.where(valid_mask_qr, err_qr, torch.full_like(err_qr, float('inf')))
        recon_pinv_clamped = torch.where(valid_mask_pinv, err_pinv, torch.full_like(err_pinv, float('inf')))
        errors = torch.minimum(recon_qr_clamped, recon_pinv_clamped)

        valid_indices_in_batch = torch.where(~torch.isinf(errors))[0]
        for idx_in_batch in valid_indices_in_batch:
            global_idx = start + idx_in_batch.item()
            batch = global_idx // n2
            seq = global_idx % n2
            err = errors[idx_in_batch].item()
            results.append((int(batch), int(seq), err))

    return results


def find_expressible_vectors3(a, b, tol_error=1e-2):
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensor")
    if torch.isnan(a).any() or torch.isinf(a).any():
        raise ValueError("Input matrix a contains NaN or Inf values.")
    if torch.isnan(b).any() or torch.isinf(b).any():
        raise ValueError("Input matrix b contains NaN or Inf values.")

    n1, n2, m = b.shape
    b_flat = b.view(-1, m)
    A = a.float()
    results = []

    try:
        q, r = torch.linalg.qr(A, mode='reduced')
        singular_values = torch.linalg.svdvals(r)
        max_sv = singular_values.max()
        rank = (singular_values > 1e-6 * max_sv).sum().item()
        if rank == 0:
            return []
        r_reduced = r[:rank, :rank]
        q_reduced = q[:, :rank]
        reg = 1e-6 * torch.eye(rank, device=A.device)
        r_reduced += reg
        batch_size_qr = 2048
        n_samples = b_flat.shape[0]
        for start in range(0, n_samples, batch_size_qr):
            end = min(start + batch_size_qr, n_samples)
            b_batch = b_flat[start:end]
            with torch.no_grad():
                try:
                    q_t_b = q_reduced.T @ b_batch.T
                    solution_qr = torch.linalg.solve_triangular(r_reduced, q_t_b, upper=True)
                    reconstructed_qr = (q_reduced @ solution_qr).T
                except Exception as e:
                    print(f"QR failed on batch {start}:{end}, error: {str(e)}")
                    valid_mask_qr = torch.zeros(b_batch.shape[0], dtype=torch.bool, device=b_batch.device)
                    err_qr = torch.full((b_batch.shape[0],), float('inf'), device=b_batch.device)
                else:
                    err_qr = torch.linalg.norm(b_batch - reconstructed_qr, dim=1)
                    valid_mask_qr = err_qr < tol_error
            with torch.no_grad():
                try:
                    pinv_A = torch.linalg.pinv(A)
                    coefficients_pinv = pinv_A @ b_batch.T
                    reconstructed_pinv = (A @ coefficients_pinv).T
                except Exception as e:
                    print(f"PINV failed on batch {start}:{end}, error: {str(e)}")
                    valid_mask_pinv = torch.zeros(b_batch.shape[0], dtype=torch.bool, device=b_batch.device)
                    err_pinv = torch.full((b_batch.shape[0],), float('inf'), device=b_batch.device)
                else:
                    err_pinv = torch.linalg.norm(b_batch - reconstructed_pinv, dim=1)
                    valid_mask_pinv = err_pinv < tol_error
            valid_mask = valid_mask_qr | valid_mask_pinv
            recon_qr_clamped = torch.where(valid_mask_qr, err_qr, torch.full_like(err_qr, float('inf')))
            recon_pinv_clamped = torch.where(valid_mask_pinv, err_pinv, torch.full_like(err_pinv, float('inf')))
            errors = torch.minimum(recon_qr_clamped, recon_pinv_clamped)
            valid_indices_in_batch = torch.where(~torch.isinf(errors))[0]
            for idx_in_batch in valid_indices_in_batch:
                global_idx = start + idx_in_batch.item()
                batch = global_idx // n2
                seq = global_idx % n2
                err = errors[idx_in_batch].item()
                results.append((int(batch), int(seq), err))
        return results
    except Exception as e:
        print(f"Batch processing failed even with partial handling: {str(e)}")
        print("Switching to per-vector fallback processing...")

    for idx in range(len(b_flat)):
        vec = b_flat[idx:idx + 1]
        try:
            q, r = torch.linalg.qr(A, mode='reduced')
            singular_values = torch.linalg.svdvals(r)
            max_sv = singular_values.max()
            rank = (singular_values > 1e-6 * max_sv).sum().item()
            if rank == 0:
                continue
            r_reduced = r[:rank, :rank]
            q_reduced = q[:, :rank]
            reg = 1e-6 * torch.eye(rank, device=A.device)
            r_reduced += reg
            q_t_b = q_reduced.T @ vec.T
            solution_qr = torch.linalg.solve_triangular(r_reduced, q_t_b, upper=True)
            reconstructed_qr = (q_reduced @ solution_qr).T
            pinv_A = torch.linalg.pinv(A)
            coefficients_pinv = pinv_A @ vec.T
            reconstructed_pinv = (A @ coefficients_pinv).T
            err_qr = torch.linalg.norm(vec - reconstructed_qr, dim=1).item()
            err_pinv = torch.linalg.norm(vec - reconstructed_pinv, dim=1).item()
            if err_qr < tol_error or err_pinv < tol_error:
                batch = idx // n2
                seq = idx % n2
                results.append((int(batch), int(seq), float(min(err_qr, err_pinv))))
        except Exception as e:
            print(f"Skipping vector at index {idx} due to error: {str(e)}")
            continue

    return results


def generate_combination(token_list1, token_list2):
    token_list1 = torch.tensor(token_list1).to(device)
    token_list2 = torch.tensor(token_list2).to(device)
    num_rows1, num_cols1 = token_list1.shape
    num_rows2, num_cols2 = token_list2.shape
    list1_expanded = token_list1.unsqueeze(1).repeat(1, num_rows2, 1)
    list2_expanded = token_list2.unsqueeze(0).repeat(num_rows1, 1, 1)
    combination = torch.cat((list1_expanded, list2_expanded), dim=2)
    return combination.view(-1, combination.shape[-1])


def generate_batch(com_bs, combination):
    num_batches = (len(combination) + com_bs - 1) // com_bs
    com_batches = [combination[i * com_bs:(i + 1) * com_bs] for i in range(num_batches)]
    return com_batches


def first_token_recon(attack_model, grade_data, com_batches):
    global attn_inputs
    possible_token = []
    for com_batch in com_batches:
        attn_inputs = []
        with torch.no_grad():
            _ = attack_model(input_ids=com_batch.clone().detach().to(device))
        attn_inputs = [tensor[:, :, :] for tensor in attn_inputs]
        attn_inputs = torch.stack(attn_inputs, dim=0)

        for layer_index in range(len(attack_model.h)):
            results = find_expressible_vectors3_preprocessed(attn_inputs[layer_index], grade_data[layer_index])
            for re_index in range(len(results)):
                possible_token.append([com_batch[results[re_index][0], results[re_index][1]].item()])
    possible_token = list(set(tuple(token) for token in possible_token))
    possible_token = [list(token) for token in possible_token]
    return possible_token


def token_recon(attack_model, grade_data, com_batches):
    global attn_inputs
    temp_possible_token = []
    for index, com_batch in enumerate(com_batches):
        attn_inputs = []
        with torch.no_grad():
            _ = attack_model(input_ids=com_batch.clone().detach().to(device))
        attn_inputs = [tensor[:, -1, :] for tensor in attn_inputs]
        attn_inputs = torch.stack(attn_inputs, dim=0).unsqueeze(2)
        for layer_index in range(1, len(model.h)):
            results = find_expressible_vectors3_preprocessed(attn_inputs[layer_index], grade_data[layer_index])
            for re_index in range(len(results)):
                temp_possible_token.append(com_batch[results[re_index][0]].tolist())
    temp_possible_token = list(set(tuple(token) for token in temp_possible_token))
    temp_possible_token = [list(token) for token in temp_possible_token]
    temp_possible_token = torch.tensor(temp_possible_token).to(device)
    return temp_possible_token


def update_token(temp_possible_token, current_possible_token, possible_token, possible_token_two, recon_sentence_one, temp_index):
    if len(temp_possible_token) == 0:
        recon_sentence_one.append(current_possible_token[temp_index])
    elif len(temp_possible_token) == 1:
        possible_token.extend(temp_possible_token.tolist())
    elif len(temp_possible_token) > 1:
        possible_token_two.extend(temp_possible_token.tolist())
    return recon_sentence_one, possible_token, possible_token_two


def prune_smallest_1_percent(attn_grade, per=0.01):
    all_abs_values = []
    for t in attn_grade:
        all_abs_values.append(t.abs().view(-1))

    all_abs_values = torch.cat(all_abs_values)

    k = int(len(all_abs_values) * per)
    if k == 0:
        k = 1
    threshold_value = torch.kthvalue(all_abs_values, k).values.item()
    print(f"Pruning threshold ({per * 100:g}% smallest): {threshold_value:.6e}")

    pruned_attn_grade = []
    for t in attn_grade:
        mask = t.abs() >= threshold_value
        pruned_t = t * mask
        pruned_attn_grade.append(pruned_t)

    return pruned_attn_grade


def append_result_json(path, batch_size, batch, recon_sentence):
    records = []
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    records.append({
        "batch_size": batch_size,
        "reference_text": batch["sentence"],
        "reconstruct_text": recon_sentence,
    })
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


randSeed = 1
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
attack_GPT_auxiliary_function.set_random_seed(randSeed)
start_attack_number = 5
attack_number = 5

model_name = "E:/Model/GPT2"
reduction_factor = 2
num_labels = 2
batch_sizes = [1]
layer_end = 12
base_dir = os.path.dirname(os.path.abspath(__file__))

data_name = 'E:/Dataset/cola_public_1.1/cola_public/raw/in_domain_train.tsv'
train_data = attack_GPT_auxiliary_function.preprocess_tsv(data_name)
train_data = attack_GPT_auxiliary_function.CustomDataset(train_data)
generator = torch.Generator().manual_seed(42)

GPT2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
GPT2_tokenizer.pad_token = GPT2_tokenizer.eos_token

model = GPT2Model.from_pretrained(model_name)
GPT_with_adapter.insert_classification_head(model, num_classes=num_labels)
model.to(device)

model_new = GPT2Model.from_pretrained(model_name)
GPT_with_adapter.insert_classification_head(model_new, num_classes=num_labels)
model_new.load_state_dict(model.state_dict())
parameters = []
for param in model_new.parameters():
    param.requires_grad = False
for param in model_new.classification_head.classifier.parameters():
    param.requires_grad = True
    parameters.append(param)
for block in model_new.h:
    for param in block.attn.c_attn.parameters():
        param.requires_grad = True
        parameters.append(param)
optimizer = torch.optim.Adam(parameters, lr=1e-5)
model_new.to(device)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

embedding_output = None
handle_hook_embedding_output = model.h[0].register_forward_hook(hook_embedding_output)
model.eval()
epsilon = 1e-20
vocab_size = len(GPT2_tokenizer)
token_indices = torch.arange(vocab_size, device=device)
vocab_size_batch = 1024
vocab_batches = [token_indices[i:i + vocab_size_batch] for i in range(0, len(token_indices), vocab_size_batch)]
Token_index_effective = []
for index_vocab_batches, vocab_batch in enumerate(vocab_batches):
    vocab_size_batch_temp = vocab_batch.clone().detach()
    vocab_size_batch_temp = vocab_size_batch_temp.unsqueeze(1)
    attention_mask = torch.ones_like(vocab_size_batch_temp)
    with torch.no_grad():
        _ = model(input_ids=vocab_size_batch_temp, attention_mask=attention_mask)
    embedding_output = embedding_output
    abs_embedding_output = torch.abs(embedding_output)
    elements_close = torch.all(abs_embedding_output < epsilon, dim=-1)
    indices_not_close = torch.where(~elements_close)[0]
    Token_index_effective.extend(vocab_batch[indices_not_close].tolist())
handle_hook_embedding_output.remove()
Token_index_effective_temp = [[token] for token in Token_index_effective]

base_dir = os.path.dirname(os.path.abspath(__file__))

per_values = [0.01, 0.5, 0.7]

for per in per_values:
    print(f"\nRunning with per = {per}")

    attn_inputs = []
    for batch_size in batch_sizes:
        print("Batch Size:", batch_size)

        result_file_save_path = os.path.join(base_dir, f'results/NAive_GP_{per}.json')

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, generator=generator)

        handle_hook_attn = []
        for block in model.h:
            handle_hook_attn_temp = block.attn.c_attn.register_forward_hook(hook_attn_inputs)
            handle_hook_attn.append(handle_hook_attn_temp)

        model_new.eval()
        for layer in model_new.h:
            layer.attn.c_attn.train()
        model_new.classification_head.train()

        rec_num_all = []
        rouge_1_all = []
        rouge_2_all = []
        rouge_l_all = []
        elapsed_time_all = []
        current_number = 0

        for batch in train_loader:
            model.load_state_dict(model_new.state_dict())
            recon_sentence_one = []
            attn_inputs = []
            optimizer.zero_grad()
            input_encodings = GPT2_tokenizer(batch["sentence"], return_tensors="pt", padding=True, truncation=True).to(device)
            logits = model_new(**input_encodings)
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()

            if current_number < start_attack_number:
                current_number = current_number + 1
                continue

            attn_grade = []
            attn_grade_bias = []
            for block in model_new.h:
                attn_grade.append(block.attn.c_attn.weight.grad)
            attn_grade = prune_smallest_1_percent(attn_grade, per=per)

            a_result = []
            for layer_index in range(len(model_new.h)):
                a_result.append(preprocess_a(attn_grade[layer_index][:, :]))

            start_time = time.time()
            possible_token = []
            possible_token_two = []
            recon_sentence_one = []

            for layer_index in range(len(model_new.h)):
                combination = generate_combination([[i] for i in range(50257)], Token_index_effective_temp)
                com_batches = generate_batch(4096, combination)
                possible_token = first_token_recon(model, a_result, com_batches)
                if len(possible_token) == 0:
                    break
                for iteration in range(3):
                    current_possible_token = possible_token
                    possible_token = []
                    for temp_index, temp_possible_token in enumerate(current_possible_token):
                        current_possible_token_two = []
                        temp_cs = 4096
                        combination = generate_combination([temp_possible_token], Token_index_effective_temp)
                        com_batches = generate_batch(temp_cs, combination)
                        temp_possible_token = token_recon(model, a_result, com_batches)
                        recon_sentence_one, possible_token, possible_token_two = update_token(temp_possible_token,
                                                                                              current_possible_token_two,
                                                                                              possible_token,
                                                                                              possible_token_two,
                                                                                              recon_sentence_one,
                                                                                              temp_index)
                    current_possible_token = possible_token_two
                    possible_token_two = []
                    if len(current_possible_token) == 0:
                        break
                    for temp_index, temp_possible_token in enumerate(current_possible_token):
                        combination = generate_combination([temp_possible_token], Token_index_effective_temp)
                        com_batches = generate_batch(temp_cs, combination)
                        temp_possible_token = token_recon(model, a_result, com_batches)
                        recon_sentence_one, possible_token, possible_token_two = update_token(temp_possible_token,
                                                                                              current_possible_token_two,
                                                                                              possible_token,
                                                                                              possible_token_two,
                                                                                              recon_sentence_one,
                                                                                              temp_index)
                    if len(current_possible_token) > 0:
                        if len(current_possible_token[0]) < 5:
                            temp_cs = 4096
                        elif 4 < len(current_possible_token[0]) < 9:
                            temp_cs = 2048
                        else:
                            temp_cs = 512
                        for temp_index, temp_possible_token in enumerate(current_possible_token):
                            combination = generate_combination([temp_possible_token], Token_index_effective_temp)
                            com_batches = generate_batch(temp_cs, combination)
                            temp_possible_token = token_recon(model, a_result, com_batches)
                            recon_sentence_one, possible_token, possible_token_two = update_token(temp_possible_token,
                                                                                                  current_possible_token,
                                                                                                  possible_token,
                                                                                                  possible_token_two,
                                                                                                  recon_sentence_one,
                                                                                                  temp_index)
                else:
                    break
            end_time = time.time()
            recon_sentence = [GPT2_tokenizer.decode(ids, clean_up_tokenization_spaces=True, skip_special_tokens=True) for ids in recon_sentence_one]

            append_result_json(result_file_save_path, batch_size, batch, recon_sentence)

            current_number = current_number + 1
            if current_number >= start_attack_number + attack_number:
                break

        for handle in handle_hook_attn:
            handle.remove()

print("Attack over!")
