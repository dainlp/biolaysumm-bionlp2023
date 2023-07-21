import argparse, numpy as np, os
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_filepath", type=str, default=None)
    parser.add_argument("--dev_filepath", type=str, default=None)
    parser.add_argument("--test_filepath", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    args = parser.parse_known_args()[0]
    return args


def main():
    args = parse_arguments()

    data_files = {}
    if args.train_filepath is not None: data_files["train"] = args.train_filepath
    if args.dev_filepath is not None: data_files["dev"] = args.dev_filepath
    if args.test_filepath is not None: data_files["test"] = args.test_filepath
    dataset = load_dataset("json", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompt_template = f"Summarize the following article:\n{{input}}\nSummary:\n"
    prompt_lenght = len(tokenizer(prompt_template.format(input=""))["input_ids"])
    max_sample_length = tokenizer.model_max_length - prompt_lenght

    tokenized_inputs = concatenate_datasets([dataset[k] for k in data_files.keys()]).map(lambda x: tokenizer(x["source"], truncation=True), batched=True, remove_columns=["source", "summary"])
    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    max_source_length = min(max_source_length, max_sample_length)
    # The maximum total sequence length for target text after tokenization.
    # Sequences longer than this will be truncated, sequences shorter will be padded."
    tokenized_targets = concatenate_datasets([dataset[k] for k in data_files.keys()]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["source", "summary"])
    # max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
    max_target_length = int(np.percentile(target_lenghts, 95))

    def preprocess_function(sample, padding="max_length"):
        # add prefix to the input for t5
        inputs = [prompt_template.format(input=s) + s for s in sample["source"]]
        # tokenize inputs
        model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length":
            labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["source", "summary", "id"])
    for split in tokenized_dataset:
        tokenized_dataset[split].save_to_disk(os.path.join(args.cache_dir, split))


if __name__ == "__main__":
    main()

