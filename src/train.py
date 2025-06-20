# train_spark_expert.py

import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

# 1. CARREGUE SEU CONJUNTO DE DADOS PERSONALIZADO
with open('spark_qa_dataset.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

contexts = [item['context'] for item in raw_data]
questions = [item['question'] for item in raw_data]
answers = [item['answers'] for item in raw_data]
dataset = Dataset.from_dict({"context": contexts, "question": questions, "answers": answers})

# 2. CARREGUE O MODELO E O TOKENIZADOR
model_checkpoint = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

# 3. PRÉ-PROCESSE OS DADOS
max_length = 384
doc_stride = 128
def preprocess_function(examples):
    tokenized_examples = tokenizer(examples["question"], examples["context"], truncation="only_second", max_length=max_length, stride=doc_stride, return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length")
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_examples.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)
    return tokenized_examples

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

# 4. AJUSTE FINO DO MODELO
training_args = TrainingArguments(
    output_dir="./spark_expert_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

print("Iniciando o treinamento final nos dados de P&R personalizados do Spark...")
trainer.train()
trainer.save_model("./spark_expert_model")
print("\nTreinamento completo. Seu modelo Especialista em Spark foi salvo.")