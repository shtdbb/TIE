import math
import torch
import torch.nn as nn
from typing import Optional, Union, Tuple, List

from transformers import BertTokenizer
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import add_code_sample_docstrings, add_start_docstrings_to_model_forward
from transformers.models.ernie.configuration_ernie import ErnieConfig


class TIE(torch.nn.Module):
    """
    Temporal Information Extraction Model
    """
    def __init__(self, encoder_config_path: str = ".", **kwargs):
        super(TIE, self).__init__(**kwargs)
        self.encoder_config = ErnieConfig.from_json_file(encoder_config_path + "/encoder_config.json")
        self.encoder = ErnieModel(self.encoder_config)
        self.begin_cls = torch.nn.Linear(self.encoder_config.hidden_size, 1)
        self.end_cls = torch.nn.Linear(self.encoder_config.hidden_size, 1)
        # 日期解码: 0~9, [U]
        self.decoder = torch.nn.Linear(self.encoder_config.hidden_size, 11)
        # self._initialize_weights()


    def forawrd(self,
                inputs_list: list[dict[str, torch.Tensor]],
                ) -> torch.Tensor:
        last_all_embedding = None
        begin_logits_list, end_logits_list, decoder_logits_list = [], [], []
        for inputs in inputs_list:
            encoder_output, cur_all_embedding = self.encoder(
                                                    input_ids=inputs["input_ids"], 
                                                    attention_mask=inputs["attention_mask"], 
                                                    token_type_ids=inputs["token_type_ids"], 
                                                    position_ids=inputs.get("position_ids", None), 
                                                    last_all_embedding=last_all_embedding)
            hidden_state = encoder_output[0]   # (bs, s, h)
            last_all_embedding = cur_all_embedding
            
            begin_logits = self.begin_cls(hidden_state)   # (bs, s, 1)
            end_logits = self.end_cls(hidden_state)   # (bs, s, 1)
            decoder_logits = self.decoder(hidden_state)   # (bs, s, 11)
            
            begin_logits_list.append(begin_logits)
            end_logits_list.append(end_logits)
            decoder_logits_list.append(decoder_logits)

        return begin_logits_list, end_logits_list, decoder_logits_list


    def _initialize_weights(self):
        """
        Initialize the weights of the begin_cls and end_cls layers using Kaiming initialization.
        """
        torch.nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        if self.begin_cls.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.decoder.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.decoder.bias, -bound, bound)


    def save_pretrained(self, save_directory: str):
        """
        Save the model to the specified directory.
        """
        # Save the entire model state dictionary
        torch.save(self.state_dict(), save_directory + "/pytorch_model_2.bin")
    


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
        


class ErniePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ErnieConfig
    base_model_prefix = "ernie"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ErnieEncoder):
            module.gradient_checkpointing = value


ERNIE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        task_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Task type embedding is a special embedding to represent the characteristic of different tasks, such as
            word-aware pre-training task, structure-aware pre-training task and semantic-aware pre-training task. We
            assign a `task_type_id` to each task and the `task_type_id` is in the range `[0,
            config.task_type_vocab_size-1]
        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

_CHECKPOINT_FOR_DOC = "nghuyong/ernie-1.0-base-zh"
_CONFIG_FOR_DOC = "ErnieConfig"

class ErnieModel(ErniePreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Ernie
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = ErnieEmbeddings(config)
        self.encoder = ErnieEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Copied from transformers.models.bert.modeling_bert.BertModel.get_input_embeddings
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # Copied from transformers.models.bert.modeling_bert.BertModel.set_input_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # Copied from transformers.models.bert.modeling_bert.BertModel._prune_heads
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(ERNIE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPoolingAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        task_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        last_all_embedding: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # 扩展维度到四维, (bs, n, s, s)
        if len(attention_mask.shape) > 4:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)
        else:
            extended_attention_mask = attention_mask
        
        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(   # (bs, s, h)
            input_ids=input_ids,   # (b, s)
            position_ids=position_ids,   # list[b, s]
            token_type_ids=token_type_ids,   # (b, s)
            task_type_ids=task_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs, cur_all_embedding = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,   # (bs, n, s, s)
            last_all_embedding=last_all_embedding,   # n_layer list * tensor(bs, h)
            head_mask=head_mask,   # (n_layers, )
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )   # output对象，存储最后一个layer的输出 hidden_states: (bs, s, h)
        sequence_output = encoder_outputs[0]   # (bs, s, h)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        ), cur_all_embedding


class ErnieOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)   # (bs, s, h)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states   # (bs, s, h)



class ErnieLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        # last chunk's [CLS] hidden states merged with current [CLS] hidden states
        self.update = nn.Linear(config.hidden_size, config.hidden_size)
        self.input = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.attention = ErnieAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = ErnieAttention(config, position_embedding_type="absolute")
        self.intermediate = ErnieIntermediate(config)
        self.output = ErnieOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (bs, s, h)
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        last_hidden_states: torch.Tensor = None,   # (bs, h)
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # long distance dependency
        if last_hidden_states is not None:
            cur_embedding = self.input(hidden_states[:, 0, :])   # (bs, h)
            last_embedding = self.update(last_hidden_states)   # (bs, h)
            hidden_states[:, 0, :] = cur_embedding + last_embedding
        
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,   # (bs, s, h)
            attention_mask,   # (bs, n, s, s)
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )   # [(bs, s, h), ]
        attention_output = self_attention_outputs[0]   # (bs, s, h)

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs   # [(bs, s, h), ]

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs   # [(bs, s, h), ]

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)   # (bs, s, 4*h)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Ernie
class ErnieEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ErnieLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        last_all_embedding: Optional[list[torch.FloatTensor]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        next_decoder_cache = () if use_cache else None
        cur_all_embedding = []
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    last_all_embedding[i] if len(last_all_embedding) > 0 else None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    last_all_embedding[i] if len(last_all_embedding) > 0 else None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )   # [(bs, s, h), ]

            hidden_states = layer_outputs[0]   # (bs, s, h)
            cur_all_embedding.append(hidden_states[:, 0, :])   # (bs, h)
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        ), cur_all_embedding


class ErnieEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)   # 40000, 768
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)    # 2048, 768
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)   # 4, 768
        self.use_task_id = config.use_task_id
        if config.use_task_id:
            self.task_type_embeddings = nn.Embedding(config.task_type_vocab_size, config.hidden_size)   # 3. 768

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)   # 768, 1e-5
        self.dropout = nn.Dropout(config.hidden_dropout_prob)   # 0.1
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))   # (1, 2048)
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )   # (1, 2048)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        task_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)   # (bs, s, h)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)   # (bs, s, h)

        embeddings = inputs_embeds + token_type_embeddings   # (bs, s, h)
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)   # (bs, s, h)
            embeddings += position_embeddings

        # add `task_type_id` for ERNIE model
        if self.use_task_id:
            if task_type_ids is None:
                task_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
            task_type_embeddings = self.task_type_embeddings(task_type_ids)   # (bs, s, h)
            embeddings += task_type_embeddings

        embeddings = self.LayerNorm(embeddings)   # (bs, s, h)
        embeddings = self.dropout(embeddings)   # (bs, s, h)
        return embeddings


class ErnieSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)   # (768, 768) + bias
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)   #   (bs, s, n, h')
        x = x.view(new_x_shape)   #   (bs, s, h) -> (bs, s, n, h')
        return x.permute(0, 2, 1, 3)   #   (bs, s, n, h') -> (bs, n, s, h')

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)   # (bs, s, h) -> (bs, s, h)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))   # (bs, n, s, h')
            value_layer = self.transpose_for_scores(self.value(hidden_states))   # (bs, n, s, h')

        query_layer = self.transpose_for_scores(mixed_query_layer)   # (bs, n, s, h')

        use_cache = past_key_value is not None
        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))   # (bs, n, s, s)

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(key_length - 1, dtype=torch.long, device=hidden_states.device).view(
                    -1, 1
                )
            else:
                position_ids_l = torch.arange(query_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(key_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)   # (bs, n, s, s)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in ErnieModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)   # (bs, n, s, s)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)   # (bs, n, s, h')

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # (bs, s, n, h')
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)   # (bs, s, h)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs   # (bs, s, h)


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput with Bert->Ernie
class ErnieSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)   # (bs, s, h)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Ernie
class ErnieAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = ErnieSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = ErnieSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,   # b, s, h
            attention_mask,   # b, n, s, s
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )   # [(bs, s, h), ]
        attention_output = self.output(self_outputs[0], hidden_states)   # (bs, s, h)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs   # # [(bs, s, h), ]


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Ernie
class ErnieIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)   # (bs, s, 4*h)
        hidden_states = self.intermediate_act_fn(hidden_states)   # (bs, s, 4*h)
        return hidden_states   # (bs, s, 4*h)

