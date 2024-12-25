from transformers import BertTokenizer
from model.ernie3_base.modeling_ernie import ErnieForMaskedLM


model_path = "/root/project/final_project/extract/model/ernie3_base"
tokenizer = BertTokenizer.from_pretrained("/root/project/final_project/extract/model/tie")
model = ErnieForMaskedLM.from_pretrained(model_path).cuda()

text = "特朗普于2021年1月20日卸任美国总统。<memory>2020年02月01日</memory>请问：特朗普的职位是美国总统的结束时间是什么时候？[Y][Y][Y][Y]年[M][M]月[D][D]日。"
inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
decode_text = tokenizer.decode(inputs["input_ids"][0])
output = model(**inputs, labels=inputs["input_ids"])
print(output.logits)   # (bs, s, vocab_size)
