import argparse, evaluate, numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from nltk.tokenize import sent_tokenize
from utils import write_object_to_json_file, write_list_to_json_file, set_seed


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_dir", type=str, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=0)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--output_metrics_filepath", type=str, default=None)
    parser.add_argument("--output_predictions_filepath", type=str, default=None)

    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fp16", default=False, action="store_true")

    parser.add_argument("--generation_max_length", type=int, default=140, help="Maximum length to use for generation")
    parser.add_argument("--generation_num_beams", type=int, default=4, help="Number of beams to use for generation.")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")

    args = parser.parse_known_args()[0]
    return args


def main():
    args = parse_arguments()
    set_seed(args.seed)
    
    test_dataset = load_from_disk(args.test_dir)
    if args.num_examples > 0:
        indices = [i for i in range(args.start_index, min(args.start_index + args.num_examples, len(test_dataset)))]
        test_dataset = test_dataset.select(indices=indices)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.output_dir, use_cache=False)

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
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=args.generation_num_beams,
        fp16=args.fp16,  # Overflows with fp16
        deepspeed=args.deepspeed,
        gradient_checkpointing=True,
        # logging & evaluation strategies
        logging_dir=args.logging_dir,
        logging_strategy="steps",
        logging_steps=500,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        # metric_for_best_model="overall_f1",
        # push to hub parameters
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    predictions = trainer.predict(test_dataset, metric_key_prefix="test")
    test_metrics = predictions.metrics
    write_object_to_json_file(test_metrics, args.output_metrics_filepath)

    test_predictions = []
    for p, g, e in zip(predictions.predictions, predictions.label_ids, test_dataset):
        p = [i for i in p if i not in [-100, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]]
        g = [i for i in g if i not in [-100, tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id]]
        p = tokenizer.decode(p)
        g = tokenizer.decode(g)
        test_predictions.append({"pred": p, "gold": g})
    write_list_to_json_file(test_predictions, args.output_predictions_filepath)


if __name__ == "__main__":
    main()

