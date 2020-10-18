'''
Author: Roy 
Date: 2020-10-18 21:50:23
LastEditTime: 2020-10-18 22:12:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /generation/models/test.py
'''
from modeling_bart import BartForConditionalGeneration
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BartTokenizer


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


if __name__ == "__main__":
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = BartForConditionalGeneration.from_pretrained(
        'facebook/bart-large')
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<xWants>', 'PersonX', 'PersonY']})
    model.resize_token_embeddings(len(tokenizer))
    head_rel = "PersonX repels PersonY's attack <xWants>"
    tail = "to defend himself"
    head_rel_tail = head_rel + " " + tail
    encoding = tokenizer.prepare_seq2seq_batch(
        head_rel, tail, max_length=20, max_target_length=20, padding='max_length', truncation=True, return_tensors='pt')
    posterior_encoding = tokenizer(head_rel_tail, add_special_tokens=True,
                                   max_length=25, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    labels = encoding['labels']
    decoder_input_ids = shift_tokens_right(labels, tokenizer.pad_token_id)
    output = model(input_ids=input_ids, attention_mask=attention_mask,
                   decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True, use_cache=False)
    posterior_output = model(
        input_ids=posterior_encoding['input_ids'], attention_mask=posterior_encoding['attention_mask'], return_dict=True)
    logits = output.logits
    loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    loss.backward()
    print(output.encoder_last_hidden_state.shape)
    print(posterior_output.encoder_last_hidden_state.shape)
    exit(0)
    for _ in range(100):
        text = "I just wanna run away."
        tgt_text = "I run away."
        input_dict = tokenizer([text], return_tensors='pt')
        decoder_input_dict = tokenizer([tgt_text], return_tensors='pt')
        outputs = model(input_ids=input_dict['input_ids'], attention_mask=input_dict['attention_mask'], decoder_input_ids=shift_tokens_right(
            decoder_input_dict['input_ids'], tokenizer.pad_token_id), use_cache=False, output_hidden_states=True, return_dict=True)
        loss = loss_fn(
            outputs.logits.view(-1, outputs.logits.shape[-1]), decoder_input_dict['input_ids'].view(-1))
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(tokenizer("I just wanna run away", return_tensors='pt')[
                                       'input_ids'], decoder_start_token_id=model.config.decoder_start_token_id, num_beams=3, max_length=10, min_length=2, early_stopping=True)
        print([tokenizer.decode(g, skip_special_tokens=True)
               for g in generated_ids])
