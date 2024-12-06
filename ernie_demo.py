from transformers import BertTokenizer
from model.ernie3_base.modeling_ernie import ErnieForMaskedLM


model_path = "model/ernie3_base"
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
model = ErnieForMaskedLM.from_pretrained(model_path).cuda()

text = "这是一个示例句子。This is a example as a sentence."
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model(**inputs, labels=inputs["input_ids"])
print(output)
