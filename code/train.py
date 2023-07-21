import argparse, evaluate, numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from nltk.tokenize import sent_tokenize
from utils import write_object_to_json_file, set_seed


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--dev_dir", type=str, default=None)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--output_metrics_filepath", type=str, default=None)
    parser.add_argument("--output_predictions_filepath", type=str, default=None)

    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")

    args = parser.parse_known_args()[0]
    return args


def main():
    args = parse_arguments()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, use_cache=False)

    train_dataset = load_from_disk(args.train_dir)
    dev_dataset = load_from_disk(args.dev_dir) if args.dev_dir is not None else None

    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=False,  # Overflows with fp16
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        # logging & evaluation strategies
        logging_dir=args.logging_dir,
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch" if dev_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if dev_dataset is not None else False,
        metric_for_best_model="rouge1",
        # push to hub parameters
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    train_outputs = trainer.train()
    trainer.save_model(args.output_dir)
    train_metrics = train_outputs.metrics
    if dev_dataset is not None:
        dev_metrics = trainer.predict(dev_dataset, metric_key_prefix="dev").metrics
        train_metrics.update(dev_metrics)
    write_object_to_json_file(train_metrics, args.output_metrics_filepath)


if __name__ == "__main__":
    main()

