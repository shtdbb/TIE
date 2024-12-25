import torch
from typing import Optional, Union

from transformers import BertTokenizer
from ..ernie3_base.modeling_ernie import ErnieModel
from transformers.models.ernie.configuration_ernie import ErnieConfig


class TIE(torch.nn.Module):
    """
    Temporal Information Extraction Model
    """
    def __init__(self, encoder_config_path: str = ".", **kwargs):
        super(TIE, self).__init__(**kwargs)
        self.encoder_config = ErnieConfig.from_json_file(encoder_config_path + "/encoder_config.json")
        self.encoder = ErnieModel(self.encoder_config, False)
        self.begin_cls = torch.nn.Linear(self.encoder_config.hidden_size, 1)
        self.end_cls = torch.nn.Linear(self.encoder_config.hidden_size, 1)


    def forawrd(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        encoder_output = self.encoder(**inputs)
        hidden_state = encoder_output[0]
        
        begin_logits = self.begin_cls(hidden_state)
        end_logits = self.end_cls(hidden_state)
        return (begin_logits, end_logits)
    


class TIETokenizer(BertTokenizer):
    def __init__(self, **kwargs):
        super(TIETokenizer, self).__init__(**kwargs)
        self.chunk_size: int = 512

    def _get_extended_attention_mask(
        self, attention_mask: torch.Tensor, input_shape: tuple[int] = None, device: torch.device = None, dtype: torch.FloatType = None
    ) -> torch.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        dtype = attention_mask.dtype
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def split_chunk(self, texts: list[str], chunk_size: Optional[int], overlap: int = 16) -> list[list[str]]:
        """
        对正文进行分段，合并的字符串段落长度不超过 chunk_size

        Args:
            text (Union[str, list[str]]): 正文
            chunk_size (int, optional): 块大小. Defaults to 512.
            overlap (int, optional): 块重叠大小. Defaults to 16.

        Returns:
            Union[list[str], list[list[str]]]: 分段后的正文
        """
        if chunk_size is None:
            chunk_size = self.chunk_size

        if isinstance(texts, str):
            texts = [texts]
        
        chunks_list = []
        for text in texts:
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)
            chunks_list.append(chunks)
        return chunks_list


    def get_inputs(self, 
                   chunks_list: list[list[str]],
                   lang: str,
                   entity_type: Optional[str] = None, 
                   head_entity: Optional[str] = None, 
                   relation_type: Optional[str] = None, 
                   tail_entity: Optional[str] = None,
                   candidate_dates: Optional[list[tuple]] = None, 
                   start_date: Optional[tuple] = None, 
                   device: str = "cpu"
                   ) -> tuple[list[dict], list[dict]]:
        """
        添加提示信息, 给切分后的每个子句构建 prompt
        * 1. 头实体抽取prompt: `[CLS]{entity_type}[SEP]{text}[SEP][PAD]...`
        * 2. 关系抽取prompt: 中文: `[CLS]{head_entity}的{relation_type}[SEP]{text}[SEP][PAD]...` 英文: `[CLS]{relation_type} of {head_entity}[SEP]{text}[SEP][PAD]...`
        * 3. 开始时间抽取prompt: 中文: `[CLS]<memory>{candidate_dates}</memory>{head_entity}的{relation_type}是{tail_entity}的开始时间是[Y][Y][Y][Y]年[M][M]月[D][D]日[SEP]{text}[SEP][PAD]...` 
        英文: `[CLS]`<memory>`{candidate_dates}</memory>{head_entity}'s {relation_type} is {tail_entity}'s start date is [Y][Y][Y][Y]-[M][M]-[D][D][SEP]{text}[SEP][PAD]...`
        * 4. 结束时间抽取prompt: 中文: `[CLS]<memory>{candidate_dates}</memory>{head_entity}的{relation_type}是{tail_entity}的开始时间是{start_date}，结束时间是[Y][Y][Y][Y]年[M][M]月[D][D]日[SEP]{text}[SEP][PAD]...` 
        英文: `[CLS]<memory>{candidate_dates}</memory>{head_entity}'s {relation_type} is {tail_entity}'s start date is {start_date}, and its end date is [Y][Y][Y][Y]-[M][M]-[D][D][SEP]{text}[SEP][PAD]...`
        其中, `candidate_dates`中的多个日期之间的attention_mask被设置为0, 以区分多个时间, 每个日期之内attention_mask设置为1

        Args:
            chunks_list (list[list[str]]): 切分后的输入，可能是一个样本的多个文段，也可能是多个样本的多个文段 
            tokenizer (PreTrainedTokenizerBase): tokenizer  
            lang (str): 语言, 中文: `zh`, 英文: `en`  
            entity_type (str, optional): 实体类型. Defaults to None.
            head_entity (str, optional): 头实体. Defaults to None.
            relation_type (str, optional): 关系类型. Defaults to None.
            tail_entity (str, optional): 尾实体. Defaults to None.
            candidate_dates (list[tuple[str]], optional): 候选日期: [('2024','01','[U][U]'), ...]. Defaults to None.  
            start_date (str, optional): 开始日期 ('2024','01','[U][U]'). Defaults to None.
            device (str, optional): 使用设备. Defaults to "cpu".

        Returns:
            tuple[list[dict], list[dict]]: inputs 列表和映射字典
        """
        # 检测语言合法性
        assert lang in ["zh", "en"], "lang must be 'zh' or 'en'"
        # 检测头实体抽取任务合法性, 如果提供了实体类型, 则其他的必须为空
        if entity_type is not None:
            assert head_entity is None, "head_entity must be None when entity_type is provided"
            assert relation_type is None, "relation_type must be None when entity_type is provided"
            assert tail_entity is None, "tail_entity must be None when entity_type is provided"
            assert candidate_dates is None, "candidate_dates must be None when entity_type is provided"
        # 检测关系抽取任务合法性
        if relation_type is not None:
            assert head_entity is not None, "head_entity must be not None when relation_type is provided"
            assert entity_type is None, "entity_type must be None when relation_type is provided"
            assert tail_entity is None, "tail_entity must be None when relation_type is provided"
            assert candidate_dates is None, "candidate_dates must be None when relation_type is provided"
        # 检测时间抽取任务合法性
        if candidate_dates is not None:
            assert head_entity is not None, "head_entity must be not None when candidate_dates is provided"
            assert relation_type is not None, "relation_type must be not None when candidate_dates is provided"
            assert tail_entity is not None, "tail_entity must be not None when candidate_dates is provided"
        # 每次只能执行一种任务
        assert (entity_type is None) + (relation_type is None) + (candidate_dates is None) == 2, "Only one task can be performed at a time"
        # 不能都为空
        assert (entity_type is not None) or (relation_type is not None) or (candidate_dates is not None), "At least one task must be performed"
        
        # 构建prompt
        inputs_list = []
        token_to_char_map = []
        
        for chunks in chunks_list:
            for chunk in chunks:
                if entity_type is not None:
                    # 头实体抽取prompt
                    prompt = f"{entity_type}[SEP]{chunk}"
                elif relation_type is not None:
                    # 关系抽取prompt
                    if lang == "zh":
                        prompt = f"{head_entity}的{relation_type}[SEP]{chunk}"
                    else:
                        prompt = f"{relation_type} of {head_entity}[SEP]{chunk}"
                elif candidate_dates is not None:
                    # 时间抽取prompt
                    
                    # 对 memory 中的数字字符单独编码
                    memory_ids = [self.convert_tokens_to_ids("<memory>")]
                    for date in candidate_dates:
                        memory_ids += \
                            [self.convert_tokens_to_ids(c) for c in date[0]] if len(date[0]) == 4 else (self.convert_tokens_to_ids(date[0])) + \
                            [self.convert_tokens_to_ids("年" if lang == "zh" else "-")] + \
                            [self.convert_tokens_to_ids(c) for c in date[1]] if len(date[1]) == 2 else (self.convert_tokens_to_ids(date[1])) + \
                            [self.convert_tokens_to_ids("月" if lang == "zh" else "-")] + \
                            [self.convert_tokens_to_ids(c) for c in date[2]] if len(date[2]) == 2 else (self.convert_tokens_to_ids(date[2])) + \
                            [self.convert_tokens_to_ids("日" if lang == "zh" else ",")]
                    memory_ids.append(self.convert_tokens_to_ids("</memory>"))
                    if lang == "zh":
                        if start_date is None:
                            # 开始时间
                            prompt = f"{head_entity}的{relation_type}是{tail_entity}的开始时间是[Y][Y][Y][Y]年[M][M]月[D][D]日[SEP]{chunk}"
                        else:
                            # 结束时间
                            prompt = f"{head_entity}的{relation_type}是{tail_entity}的开始时间是{start_date[0]}年{start_date[1]}月{start_date[2]}日，结束时间是[Y][Y][Y][Y]年[M][M]月[D][D]日[SEP]{chunk}"
                    else:
                        if start_date is None:
                            prompt = f"{head_entity}'s {relation_type} is {tail_entity}'s start date is [Y][Y][Y][Y]-[M][M]-[D][D][SEP]{chunk}"
                        else:
                            prompt = f"{head_entity}'s {relation_type} is {tail_entity}'s start date is {start_date[0]}-{start_date[1]}-{start_date[2]}, and its end date is [Y][Y][Y][Y]-[M][M]-[D][D][SEP]{chunk}"
            
                # Tokenize the prompt
                inputs = super(TIETokenizer, self).__call__(prompt, return_tensors="pt", truncation=True, padding="max_length", 
                                                              max_length=self.model_max_length).to(device)
                
                
                if candidate_dates is not None:
                    # a. 拼接记忆
                    inputs.input_ids = torch.cat([
                                                            inputs.input_ids[..., 0], 
                                                            torch.tensor([memory_ids]).to(device), 
                                                            inputs.input_ids[..., 1: ]
                                                        ], dim=-1)
                    inputs.attention_mask = torch.cat([
                                                            inputs.attention_mask[..., 0], 
                                                            torch.tensor([1] * len(memory_ids)).to(device), 
                                                            inputs.attention_mask[..., 1: ]
                                                        ], dim=-1)
                    inputs.token_type_ids = torch.cat([
                                                            inputs.token_type_ids[..., 0], 
                                                            torch.tensor([1] * len(memory_ids)).to(device), 
                                                            inputs.token_type_ids[..., 1: ]
                                                        ], dim=-1)
                    
                    # b. 给 memory 的日期之间相互 mask
                    memory_start = 1
                    memory_end = inputs.input_ids[0].tolist().index(self.convert_tokens_to_ids("</memory>"))
                    extended_attention_mask = self._get_extended_attention_mask(inputs.attention_mask)   # (1, 1, seq, seq)
                    date_start = memory_start + 1
                    for date in candidate_dates:
                        date_tokens = self.tokenize(date)
                        date_len = len(date_tokens)
                        date_end = date_start + date_len - 1
                        extended_attention_mask[..., date_start: date_end+1, date_end+1: memory_end] = 0
                        date_start = date_end + 1
                    
                    # c. 修改位置编码
                    position_ids = list(range(len(inputs.input_ids[0])))
                    position_ids[2: 2 + 11 * len(candidate_dates)] = list(range(2, 13)) * len(candidate_dates)
                    inputs["position_ids"] = [position_ids]
                    
                inputs_list.append(inputs)
                
                # 生成token位置和chunk字符串的映射
                input_ids = inputs.input_ids[0].tolist()
                cur_map: dict[int, list[int]] = {}
                current_char_start = 0
                sep_count = 0
                for i, token_id in enumerate(input_ids):
                    token = self.convert_ids_to_tokens(token_id)
                    if token == self.cls_token:
                        cur_map[i] = [-1, -1]
                    elif token == self.sep_token:
                        cur_map[i] = [-1, -1]
                        sep_count += 1
                        current_char_start = 0  # Reset char start after each SEP
                    else:
                        if sep_count == 1:  # Only start recording after the first SEP
                            token_length = len(self.convert_tokens_to_string([token]))
                            cur_map[i] = [current_char_start, current_char_start + token_length - 1]
                            current_char_start += token_length
                        else:
                            cur_map[i] = [-1, -1]
                token_to_char_map.append(cur_map)
        
        return inputs_list, token_to_char_map
        

        

