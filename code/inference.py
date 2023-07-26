import argparse, json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", type=str, default="/scratch/ik70/Corpora/flan-t5/flan-t5-xxl")
    parser.add_argument("--input_filepath", type=str, default="/home/599/xd2744/2311AC/data/sample.json")
    parser.add_argument("--max_input_length", type=int, default=1024)

    args = parser.parse_known_args()[0]
    return args


def main():
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, use_cache=False).half()
    model = model.to("cuda")

    for i, line in enumerate(open(args.input_filepath)):
        line = json.loads(line)
        prompt = f"Summarize the following article:\n{line['source']}\nSummary:\n"
        print("*" * 50)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:,:args.max_input_length]
        input_ids = input_ids.to("cuda")
        outputs = model.generate(input_ids)
        outputs = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"{i + 1}....")
        print(outputs)


if __name__ == "__main__":
    main()

