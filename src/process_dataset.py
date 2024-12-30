import os
import json
import time
import requests
from glob import glob
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
logger.add("logs/process_dataset.log", rotation="500 MB", compression="zip", enqueue=True)

DATE_PROMPT = """你是一名资深的信息抽取专家，负责对一段文本中涉及到的知识图谱三元组（头实体、关系、尾实体）进行时间抽取。

【要求】
1. 你需要根据给定的一个正文和已抽取出的多个三元组，对三元组中的事实进行开始时间和结束时间的抽取。
2. 你必须以json格式进行输出，output字段的值list的每个元素为对于的一个三元组的输出。每个三元组只有两个字段："start_date"和"end_date"，每个字段的值为日期字符串"yyyy-mm-dd"，若无法从正文中得到日期信息，则用[U]占位。格式示例：{{"output": [{{"start_date": "2024-01-11", "end_date": "[U][U][U][U]-02-02"}}, {{"start_date": "1998-11-02", "end_date": "[U][U][U][U]-[U][U]-[U][U]"}}]}}。
3. 给定的多个三元组表示(头实体, 关系, 尾实体)，如：(Elon Musk, founded, SpaceX)，表示Elon Musk创建了SpaceX。

【示例】
正文：Tech Giant Inc. has unveiled the "SmartHome AI Hub," a device that uses artificial intelligence to control home appliances, lighting, and security systems, set to launch in December 2023.
三元组：(Tech Giant Inc., Unveils, SmartHome AI Hub), (SmartHome AI Hub, Controls, Home Appliances), (SmartHome AI Hub, Launch Date, December 2023)
输出：{{"output": [{{"start_date": "[U][U][U][U]-[U][U]-[U][U]", "end_date": "[U][U][U][U]-[U][U]-[U][U]"}}, {{"start_date": "2023-12-[U][U]", "end_date": "[U][U][U][U]-[U][U]-[U][U]"}}, {{"start_date": "[U][U][U][U]-[U][U]-[U][U]", "end_date": "2023-12-[U][U]"}}]}}

【真实数据】
正文：{content}
三元组：{triplets}
输出："""


CHECK_PROMPT = """你是一名资深的信息核对专家，负责对一段文本中的知识图谱三元组（头实体、关系、尾实体）进行其发生和结束日期的核对。

【要求】
1. 你需要根据给定的一个正文和已抽取出的多个三元组和对应日期（分别为开始日期和结束日期），对时间信息是否正确进行核对。时间中的[U]表示没有提及时间；
2. 你必须以json格式进行输出，对信息的总体正确性进行判定，字段有："is_correct"，1表示正确，0表示错误；"why"，简短的解释错误原因。如：{{"is_correct": 0, "why": "没有提到正文发布时间"}}。
3. 时间没提及或者时间提及比较模糊的都可以算对，只有正文准确提及时间且三元组时间不匹配的才算错。

【示例】
正文：Tech Giant Inc. has unveiled the "SmartHome AI Hub," a device that uses artificial intelligence to control home appliances, lighting, and security systems, set to launch in December 2023.
三元组：(Tech Giant Inc., Unveils, SmartHome AI Hub, [U][U][U][U]-[U][U]-[U][U], [U][U][U][U]-[U][U]-[U][U]), (SmartHome AI Hub, Controls, Home Appliances, 2023-12-[U][U], [U][U][U][U]-[U][U]-[U][U]), (SmartHome AI Hub, Launch Date, December 2023, [U][U][U][U]-[U][U]-[U][U], 2023-12-[U][U])
输出：{{"is_correct": 1, "why": "无错误原因"}}

【真实数据】
正文：{content}
三元组：{triplets}
输出："""


URL_1 = os.getenv("DEEPSEEK_V3_URL")
KEY_1 = os.getenv("DEEPSEEK_V3_KEY")

URL_2 = os.getenv("BASE_URL")
KEY_2 = os.getenv("BASE_KEY")

def load_json(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.loads(f.read().strip())
    logger.info(f"Load {len(data)} samples from `{file_path}`")
    return data


def generate_date(content: str, triplets: list[tuple[str]], n=3, reason=None, date_tuple=None) -> tuple[str]:
    prompt = DATE_PROMPT.format(content=content, triplets=", ".join(["(\"" + "\", \"".join(t)+ "\")" for t in triplets]))
    if reason is None:
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                'type': 'json_object'
            }
        }
    else:
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}, 
                         {"role": "assistant", "content": str({"output": list(date_tuple)})},
                         {"role": "user", "content": "输出有问题，请修改。" + reason}
                         ],
            "response_format": {
                'type': 'json_object'
            }
        }

    head = {"Authorization": f"Bearer {KEY_1}", "Content-Type": "application/json"}
    response = requests.post(
        url=URL_1,
        json=data,
        headers=head
    )

    while n > 0:
        try:
            date_list = json.loads(response.json()["choices"][0]["message"]["content"])["output"]
            if len(date_list) != len(triplets):
                data["messages"][0]["content"] += f"（请注意生成三元组相应数量的{len(triplets)}个结果，不要遗漏）"
                response = requests.post(url=URL_1, json=data, headers=head)
                date_list = json.loads(response.json()["choices"][0]["message"]["content"])["output"]
                assert len(date_list) == len(triplets), f"date_list length `{len(date_list)}` not equal to triplets length `{len(triplets)}`, response: {date_list}"
            output = tuple(date_list)
            break
        except Exception as e:
            logger.warning(f"{e}, in {'first' if reason is None else 'second'} generation the {4-n} time(s) retrying...\nresponse: {response.json()}\n\n")
            n -= 1
            output = None
            time.sleep(1)
    return output


def check_date(content: str, triplets, date_tuple, n = 3) -> bool:
    prompt = CHECK_PROMPT.format(content=content, triplets=", ".join(["(\"" + "\", \"".join(t + d) + "\")" for t, d in zip(triplets, date_tuple)]))
    
    data = {
        "model": "deepseek-ai/DeepSeek-V2.5",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            'type': 'json_object'
        }
    }
    head = {"Authorization": f"Bearer {KEY_2}", "Content-Type": "application/json"}
    
    while n > 0:
        try:
            response = requests.post(
                url=URL_2,
                json=data,
                headers=head
            )
            response = json.loads(response.json()["choices"][0]["message"]["content"])
            assert response.get("is_correct") is not None
            break
        except Exception as e:
            logger.warning(f"{e}, the {4-n} time(s) retrying...\nresponse: {response.json()}\n\n")
            n -= 1
            response = None
            time.sleep(1)
    return response


def create_data(content: str, triplets: list[tuple[str]]) -> tuple[str]:
    date_tuple = generate_date(content, triplets)
    assert date_tuple is not None, "date_tuple is None"
    logger.info(f"first date: {date_tuple}")
    check_res = check_date(content, triplets, ((d["start_date"], d["end_date"]) for d in date_tuple))
    assert check_res is not None, "check_res is None"
    logger.info(f"check result: {check_res}")

    if check_res.get("is_correct") == 0:
        logger.info("check not pass, generate date again")
        date_tuple = generate_date(content, triplets, reason=check_res.get("why"), date_tuple=date_tuple)
        logger.info(f"second date: {date_tuple}")

    assert date_tuple is not None, "date_tuple is None"
    return [[d["start_date"], d["end_date"]] for d in date_tuple]


def convert_format(data_paths: list[str], dataset_type: str):
    if dataset_type == "duie":
        # 把数据集的每行例如：{"text": "《邪少兵王》是冰火未央写的网络小说连载于旗峰天下", "spo_list": [{"predicate": "作者", 
        # "object_type": {"@value": "人物"}, "subject_type": "图书作品", "object": {"@value": "冰火未央"}, "subject": "邪少兵王"}]}
        # 转为：{"tokens": ["Intravenous", "azithromycin", "-", "induced", "ototoxicity", "."], 
        # "entities": [{"type": "Adverse-Effect", "start": 4, "end": 5}, {"type": "Drug", "start": 1, "end": 2}], 
        # "relations": [{"type": "Adverse-Effect", "head": 0, "tail": 1}]}
        for path in data_paths:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            datas = []
            for line in tqdm(lines, desc="convert " + path):
                item = json.loads(line.strip())
                text = item["text"].replace("\u3000", " ")
                tokens = list(text)
                relations = item["spo_list"]
                new_ents = []
                new_rels = []
                entities = {}   # {ent_name: idx}
                for rel in relations:
                    h = rel["subject"].replace("\u3000", " ")
                    if h not in entities:
                        h_type = rel["subject_type"]
                        h_start = text.index(h)
                        h_end = h_start + len(h)
                        new_ents.append({"type": h_type, "start": h_start, "end": h_end})
                        entities[h] = len(entities)
                        
                    r = rel["predicate"].replace("\u3000", " ")
                    t = rel["object"]["@value"].replace("\u3000", " ")
                    if t not in entities:
                        t_start = text.index(t)
                        t_end = t_start + len(t)
                        t_type = rel["object_type"]["@value"]
                        new_ents.append({"type": t_type, "start": t_start, "end": t_end})
                        entities[t] = len(entities)
                        
                    new_rels.append({"type": r, "head": entities[h], "tail": entities[t]})
                
                datas.append({"tokens": tokens, "entities": new_ents, "relations": new_rels})
                
            document_path = "/".join(path.split("/")[: -1]) + "_processed"
            # 判断路径是否存在
            if not os.path.exists(document_path):
                os.makedirs(document_path)
            # 保存json文件
            with open(document_path + "/" + path.split("/")[-1], "w", encoding="utf-8") as new:
                json.dump(datas, new, ensure_ascii=False)


if __name__ == "__main__":
    files = [# "data/CONLL04_processed/conll04_train.json", "data/CONLL04_processed/conll04_test.json", "data/CONLL04_processed/conll04_dev.json", 
            #  "data/ADE_procesed/ade_full.json", 
            #  "data/SCIERC_processed/scierc_train.json", "data/SCIERC_processed/scierc_dev.json", 
            #  "data/SCIERC_processed/scierc_test.json",
             "data/DuIE2_processed/train.json", "data/DuIE2_processed/dev.json"]

    for file_path in files:
        datas = load_json(file_path)
        rel_map = load_json(glob("/".join(file_path.split("/")[: -1]) + "/*_types.json")[0])["relations"]
        for idx, data in enumerate(tqdm(datas, desc=file_path.split('/')[-1])): 
            # if idx < 56: continue

            logger.info(f"[{file_path}] idx {idx} is beginning ...")
            content = " ".join(data["tokens"]) if "DuIE" not in file_path else "".join(data["tokens"])
            entities = []
            triplets = []
            for i, t in enumerate(data["entities"]):
                e = " ".join(data["tokens"][t["start"]: t["end"]]) if "DuIE" not in file_path else "".join(data["tokens"][t["start"]: t["end"]])
                data["entities"][i]["name"] = e
                entities.append(e)
            for r in data["relations"]:
                triplets.append((entities[r["head"]], rel_map[r["type"]]["verbose"], entities[r["tail"]]))

            while True:
                try:
                    output = create_data(content, triplets)
                    break
                except Exception as e:
                    logger.warning(f"[{file_path}] idx {idx} is error, {e}, retrying ...")
                    time.sleep(1)

            for i, d in enumerate(output):
                data["relations"][i]["start_date"] = d[0]
                data["relations"][i]["end_date"] = d[1]
            
            with open(file_path.replace(".json", "_1.jsonl"), "a") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            logger.info(f"[{file_path}] idx {idx} is finished.")
    
    
    # files = ["data/DuIE2/train.json", "data/DuIE2/dev.json"]
    # convert_format(files, "duie")
