from transformers import BertTokenizer

model_path = "model/ernie3_base"
tokenizer = BertTokenizer.from_pretrained(model_path)

tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
special_tokens = ["[Y]", "[M]", "[D]", "[U]", "<memory>", "</memory>"]
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

tokenizer.save_pretrained("model/tie")
