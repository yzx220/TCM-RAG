# -*- coding: utf-8 -*-
import os
import jieba
import dataclasses as dc
import functools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Optional, Union
import numpy as np
import ruamel.yaml as yaml
import torch
import typer
from datasets import Dataset, DatasetDict, NamedSplit, Split, load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import nltk
from peft import (
    PeftConfig,
    PeftModelForCausalLM,
    get_peft_config,
    get_peft_model
)
from rouge_chinese import Rouge
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Seq2SeqTrainingArguments, AutoConfig,
)
from transformers import DataCollatorForSeq2Seq as _DataCollatorForSeq2Seq

from transformers import Seq2SeqTrainer as _Seq2SeqTrainer
import torch.nn.functional as F

ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
app = typer.Typer(pretty_exceptions_show_locals=False)


class DataCollatorForSeq2Seq(_DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        output_ids = (
            [feature['output_ids'] for feature in features]
            if 'output_ids' in features[0].keys()
            else None
        )
        if output_ids is not None:
            max_output_length = max(len(out) for out in output_ids)
            if self.pad_to_multiple_of is not None:
                max_output_length = (
                        (
                                max_output_length + self.pad_to_multiple_of - 1) //
                        self.pad_to_multiple_of * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.tokenizer.pad_token_id] * (
                        max_output_length - len(feature['output_ids'])
                )
                if isinstance(feature['output_ids'], list):
                    feature['output_ids'] = feature['output_ids'] + remainder
                else:
                    feature['output_ids'] = np.concatenate(
                        [feature['output_ids'], remainder]
                    ).astype(np.int64)
        return super().__call__(features, return_tensors)


class TCMLoss:
    def __init__(self, num_tcm_categories=4):
        self.num_tcm_categories = num_tcm_categories
        
    def calculate_tcm_entity_loss(self, logits, labels):
        """计算TCM实体分类的损失 Lc"""
        return -torch.sum(torch.log(torch.softmax(logits, dim=-1)) * labels) / len(labels)
    
    def calculate_attribute_loss(self, pred_attrs, true_attrs):
        """计算TCM属性的交叉熵损失 Lattr"""
        total_loss = 0
        # k=1: 性质(寒热温凉平)
        nature_loss = F.cross_entropy(pred_attrs['nature'], true_attrs['nature'])
        # k=2: 五味(酸苦甘辛咸)
        flavor_loss = F.cross_entropy(pred_attrs['flavor'], true_attrs['flavor'])
        # k=3: 归经
        meridian_loss = F.binary_cross_entropy_with_logits(pred_attrs['meridian'], true_attrs['meridian'])
        # k=4: 配伍禁忌
        compatibility_loss = F.binary_cross_entropy_with_logits(pred_attrs['compatibility'], true_attrs['compatibility'])
        
        total_loss = nature_loss + flavor_loss + meridian_loss + compatibility_loss
        return total_loss
        
    def calculate_safety_loss(self, response_logits, context, penalty_weight=1.0):
        """计算安全响应生成的损失 Lr"""
        # 检测禁忌组合
        Ldh = self.detect_forbidden_combinations(response_logits, context)
        # 计算理论合规性损失
        response_loss = -torch.log(torch.softmax(response_logits, dim=-1))
        # 合并损失
        return response_loss + penalty_weight * Ldh
        
    def detect_forbidden_combinations(self, response, context):
        """检测是否存在禁忌组合
        
        Args:
            response: 模型生成的响应logits
            context: 输入上下文的token ids
            
        Returns:
            torch.Tensor: 禁忌组合的惩罚值
        """
        # 定义常见的中药禁忌组合
        forbidden_pairs = {
            "附子": ["瓜蒌", "贝母", "半夏", "天花粉", "白蔹", "白及"],
            "乌头": ["瓜蒌", "贝母", "半夏", "天花粉", "白蔹", "白及"],
            "川乌": ["瓜蒌", "贝母", "半夏", "天花粉", "白蔹", "白及"],
            "草乌": ["瓜蒌", "贝母", "半夏", "天花粉", "白蔹", "白及"],
            "天南星": ["咸味药"],
            "半夏": ["咸味药"],
            "白附子": ["咸味药"],
            "甘遂": ["海藻", "瓜蒌"],
            "芫花": ["海藻", "瓜蒌"],
            "大戟": ["海藻", "瓜蒌"],
        }
        
        # 将logits转换为token ids
        pred_tokens = torch.argmax(response, dim=-1)
        
        # 解码预测的文本和上下文
        pred_text = self.tokenizer.decode(pred_tokens)
        context_text = self.tokenizer.decode(context)
        
        # 检查禁忌组合
        penalty = 0.0
        for herb1, forbidden_list in forbidden_pairs.items():
            if herb1 in pred_text or herb1 in context_text:
                for herb2 in forbidden_list:
                    if herb2 in pred_text or herb2 in context_text:
                        penalty += 1.0
                        
        # 检查特殊规则
        if "附子" in pred_text and "生用" in pred_text:
            penalty += 1.0
        if "天南星" in pred_text and "生用" in pred_text:
            penalty += 1.0
            
        return torch.tensor(penalty, device=response.device)


class TCMMetrics:
    def __init__(self):
        self.rouge = Rouge()
        # 确保下载必要的NLTK数据
        try:
            nltk.data.find('wordnet')
        except LookupError:
            nltk.download('wordnet')
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
        
    def compute_metrics(self, pred: EvalPrediction) -> dict:
        """计算TCM特定的评估指标，包括ROUGE、BLEU和METEOR
        
        Args:
            pred: 包含预测结果和标签的EvalPrediction对象
            
        Returns:
            包含各项评估指标的字典
        """
        predictions = pred.predictions
        labels = pred.label_ids
        
        # 解码预测和标签
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 计算ROUGE分数
        rouge_scores = self.rouge.get_scores(decoded_preds, decoded_labels, avg=True)
        
        # 计算BLEU分数
        bleu_scores = {
            'bleu-1': [],
            'bleu-2': [],
            'bleu-3': [],
            'bleu-4': [],
        }
        
        # 计算METEOR分数
        meteor_scores = []
        
        # 对每个预测-标签对计算分数
        for pred, label in zip(decoded_preds, decoded_labels):
            # 中文分词
            pred_tokens = list(jieba.cut(pred))
            label_tokens = list(jieba.cut(label))
            
            # 计算不同n-gram的BLEU分数
            for n in range(1, 5):
                weights = tuple([1./n] * n + [0.] * (4-n))
                score = sentence_bleu(
                    [label_tokens], 
                    pred_tokens,
                    weights=weights,
                    smoothing_function=SmoothingFunction().method1
                )
                bleu_scores[f'bleu-{n}'].append(score)
            
            # 计算METEOR分数（使用英文分词方式，因为METEOR主要用于英文）
            # 对于中文，我们先将分词结果转换为空格分隔的字符串
            pred_str = ' '.join(pred_tokens)
            label_str = ' '.join(label_tokens)
            meteor = meteor_score([word_tokenize(label_str)], word_tokenize(pred_str))
            meteor_scores.append(meteor)
        
        # 合并所有评估指标
        metrics = {
            "rouge-1": rouge_scores["rouge-1"]["f"],
            "rouge-2": rouge_scores["rouge-2"]["f"],
            "rouge-l": rouge_scores["rouge-l"]["f"],
        }
        
        # 添加BLEU分数
        for n in range(1, 5):
            metrics[f'bleu-{n}'] = np.mean(bleu_scores[f'bleu-{n}'])
            
        # 添加METEOR分数
        metrics['meteor'] = np.mean(meteor_scores)
        
        # 计算语料级别的BLEU分数
        references = [[list(jieba.cut(label))] for label in decoded_labels]
        hypotheses = [list(jieba.cut(pred)) for pred in decoded_preds]
        corpus_bleu_score = corpus_bleu(
            references,
            hypotheses,
            smoothing_function=SmoothingFunction().method1
        )
        metrics['corpus-bleu'] = corpus_bleu_score
        
        return metrics


class Seq2SeqTrainer(_Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tcm_loss = TCMLoss()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """重写计算损失函数，加入TCM特定的损失"""
        outputs = model(**inputs)
        
        # 原始语言模型损失
        original_loss = outputs.loss
        
        # 添加TCM特定的损失
        tcm_entity_loss = self.tcm_loss.calculate_tcm_entity_loss(
            outputs.logits, inputs.get("tcm_labels", None)
        )
        
        # 属性损失
        attr_loss = self.tcm_loss.calculate_attribute_loss(
            outputs.get("tcm_attrs", {}), 
            inputs.get("tcm_attr_labels", {})
        )
        
        # 安全响应损失
        safety_loss = self.tcm_loss.calculate_safety_loss(
            outputs.logits, 
            inputs["input_ids"]
        )
        
        # 合并所有损失
        total_loss = original_loss + tcm_entity_loss + attr_loss + safety_loss
        
        return (total_loss, outputs) if return_outputs else total_loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: dict[str, Any],
            prediction_loss_only: bool,
            ignore_keys=None,
            **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.args.predict_with_generate:
            output_ids = inputs.pop('output_ids')
        input_ids = inputs['input_ids']
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs
        )
        generated_tokens = generated_tokens[:, input_ids.size()[1]:]
        if self.args.predict_with_generate:
            labels = output_ids
        return loss, generated_tokens, labels


def _resolve_path(path: Union[str, Path]) -> Path:
    return Path(path).expanduser().resolve()


def _sanity_check(
        input_ids: Sequence[int],
        output_ids: Sequence[int],
        tokenizer: PreTrainedTokenizer,
):
    print('--> Sanity check')
    for in_id, out_id in zip(input_ids, output_ids):
        if in_id == 0:
            continue
        if in_id in tokenizer.tokenizer.index_special_tokens:
            in_text = tokenizer.tokenizer.index_special_tokens[in_id]
        else:
            in_text = tokenizer.decode([in_id])
        print(f'{repr(in_text):>20}: {in_id} -> {out_id}')


@functools.cache
def _get_yaml_parser() -> yaml.YAML:
    parser = yaml.YAML(typ='safe', pure=True)
    parser.indent(mapping=2, offset=2, sequence=4)
    parser.default_flow_style = False
    return parser


@dc.dataclass
class DataConfig(object):
    train_file: str
    val_file: Optional[str] = None
    test_file: Optional[str] = None

    num_proc: Optional[int] = None

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dc.dataclass
class FinetuningConfig(object):
    data_config: DataConfig

    max_input_length: int
    max_output_length: int

    training_args: Seq2SeqTrainingArguments = dc.field(
        default_factory=lambda: Seq2SeqTrainingArguments(output_dir='./output')
    )
    peft_config: Optional[PeftConfig] = None

    def __post_init__(self):
        if not self.training_args.do_eval or self.data_config.val_file is None:
            # skips the evaluation stage when `do_eval` or `eval_file` is not provided
            self.training_args.do_eval = False
            self.training_args.evaluation_strategy = 'no'
            self.data_config.val_file = None
        else:
            self.training_args.per_device_eval_batch_size = (
                    self.training_args.per_device_eval_batch_size
                    or self.training_args.per_device_train_batch_size
            )

    @classmethod
    def from_dict(cls, **kwargs) -> 'FinetuningConfig':
        training_args = kwargs.get('training_args', None)
        if training_args is not None and not isinstance(
                training_args, Seq2SeqTrainingArguments
        ):
            gen_config = training_args.get('generation_config')
            # TODO: a bit hacky
            if not isinstance(gen_config, GenerationConfig):
                training_args['generation_config'] = GenerationConfig(
                    **gen_config
                )
            kwargs['training_args'] = Seq2SeqTrainingArguments(**training_args)

        data_config = kwargs.get('data_config')
        if not isinstance(data_config, DataConfig):
            kwargs['data_config'] = DataConfig(**data_config)

        peft_config = kwargs.get('peft_config', None)
        if peft_config is not None and not isinstance(peft_config, PeftConfig):
            kwargs['peft_config'] = get_peft_config(peft_config)
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> 'FinetuningConfig':
        path = _resolve_path(path)
        kwargs = _get_yaml_parser().load(path)
        return cls.from_dict(**kwargs)


def _load_datasets(
        data_dir: Path,
        data_format: str,
        data_files: dict[NamedSplit, str],
        num_proc: Optional[int],
) -> DatasetDict:
    if data_format in ('.csv', '.json', '.jsonl'):
        dataset_dct = load_dataset(
            data_format[1:],
            data_dir=data_dir,
            data_files=data_files,
            num_proc=num_proc,
        )
    else:
        err_msg = f"Cannot load dataset in the '{data_format}' format."
        raise NotImplementedError(err_msg)

    return dataset_dct


class DataManager(object):
    def __init__(self, data_dir: str, data_config: DataConfig):
        self._num_proc = data_config.num_proc

        self._dataset_dct = _load_datasets(
            _resolve_path(data_dir),
            data_config.data_format,
            data_config.data_files,
            self._num_proc,
        )

    def _get_dataset(self, split: NamedSplit) -> Optional[Dataset]:
        return self._dataset_dct.get(split, None)

    def get_dataset(
            self,
            split: NamedSplit,
            process_fn: Callable[[dict[str, Any]], dict[str, Any]],
            batched: bool = True,
            remove_orig_columns: bool = True,
    ) -> Optional[Dataset]:
        orig_dataset = self._get_dataset(split)
        if orig_dataset is None:
            return

        if remove_orig_columns:
            remove_columns = orig_dataset.column_names
        else:
            remove_columns = None
        return orig_dataset.map(
            process_fn,
            batched=batched,
            remove_columns=remove_columns,
            num_proc=self._num_proc,
        )


def print_model_size(model: PreTrainedModel):
    print("--> Model")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> model has {total_params / 1e6}M params\n")


def process_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    batched_labels = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids, loss_masks = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ], [False, False]

        if tools is not None:
            # 处理工具相关的信息
            tool_ids = []
            for tool in tools:
                # 添加工具名称
                tool_name = f"<|{tool['name']}|>"
                tool_name_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tool_name))
                tool_ids.extend(tool_name_ids)
                
                # 添加工具描述
                if 'description' in tool:
                    tool_desc_ids = tokenizer.encode(tool['description'], add_special_tokens=False)
                    tool_ids.extend(tool_desc_ids)
                
                # 添加工具参数
                if 'parameters' in tool:
                    params_text = str(tool['parameters'])
                    params_ids = tokenizer.encode(params_text, add_special_tokens=False)
                    tool_ids.extend(params_ids)
                
            input_ids.extend(tool_ids)
            loss_masks.extend([False] * len(tool_ids))

        for message in conv:
            if message['role'] in ('system', 'user'):
                loss_mask_val = False
            else:
                loss_mask_val = True

            if message['role'] == 'tool':
                # 处理工具调用结果
                tool_result_ids = tokenizer.encode(
                    f"<|tool_response|>{message['content']}", 
                    add_special_tokens=False
                )
                new_input_ids = tool_result_ids
                new_loss_masks = [False] * len(tool_result_ids)
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                new_loss_masks = [loss_mask_val] * len(new_input_ids)

            input_ids += new_input_ids
            loss_masks += new_loss_masks

        input_ids.append(tokenizer.eos_token_id)
        loss_masks = [False, *loss_masks]
        labels = []
        for input_id, mask in zip(input_ids, loss_masks):
            if mask:
                labels.append(input_id)
            else:
                labels.append(-100)
        max_length = max_input_length + max_output_length + 1
        batched_input_ids.append(input_ids[:max_length])
        batched_labels.append(labels[:max_length])
    return {'input_ids': batched_input_ids, 'labels': batched_labels}


def process_batch_eval(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    batched_tools = batch.get('tools', None)
    batched_conv = batch['conversations']
    batched_input_ids = []
    # To avoid computing loss, we do not provide the `labels` field in the input dictionary.
    batched_output_ids = []

    if batched_tools is None:
        batched_tools = [None] * len(batched_conv)

    for tools, conv in zip(batched_tools, batched_conv):
        input_ids = [
            tokenizer.get_command('[gMASK]'),
            tokenizer.get_command('sop'),
        ]

        if tools is not None:
            # 处理工具相关的信息
            tool_ids = []
            for tool in tools:
                # 添加工具名称
                tool_name = f"<|{tool['name']}|>"
                tool_name_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(tool_name))
                tool_ids.extend(tool_name_ids)
                
                # 添加工具描述
                if 'description' in tool:
                    tool_desc_ids = tokenizer.encode(tool['description'], add_special_tokens=False)
                    tool_ids.extend(tool_desc_ids)
                
                # 添加工具参数
                if 'parameters' in tool:
                    params_text = str(tool['parameters'])
                    params_ids = tokenizer.encode(params_text, add_special_tokens=False)
                    tool_ids.extend(params_ids)
                
            input_ids.extend(tool_ids)

        for message in conv:
            if len(input_ids) >= max_input_length:
                break
            if message['role'] == 'tool':
                # 处理工具调用结果
                tool_result_ids = tokenizer.encode(
                    f"<|tool_response|>{message['content']}", 
                    add_special_tokens=False
                )
                input_ids.extend(tool_result_ids)
            else:
                new_input_ids = tokenizer.build_single_message(
                    message['role'], '', message['content']
                )
                if message['role'] == 'assistant':
                    output_prompt, output_ids = (
                        new_input_ids[:1],
                        new_input_ids[1:],
                    )
                    output_ids.append(tokenizer.eos_token_id)
                    batched_input_ids.append(
                        input_ids[:max_input_length] + output_prompt[:1]
                    )
                    batched_output_ids.append(output_ids[:max_output_length])
                input_ids += new_input_ids
    return {'input_ids': batched_input_ids, 'output_ids': batched_output_ids}


def _prepare_model_for_training(model: nn.Module, use_cpu: bool):
    for param in model.parameters():
        if param.requires_grad or use_cpu:
            param.data = param.data.to(torch.float32)


def load_tokenizer_and_model(
        model_dir: str,
        peft_config: Optional[PeftConfig] = None,
) -> tuple[PreTrainedTokenizer, nn.Module]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if peft_config is not None:
        if peft_config.peft_type.name == "PREFIX_TUNING":
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            config.pre_seq_len = peft_config.num_virtual_tokens
            config.use_cache = False
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                config=config,
            )
        if peft_config.peft_type.name == "LORA":
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                trust_remote_code=True,
                empty_init=False,
                use_cache=False
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            empty_init=False,
            use_cache=False
        )
    print_model_size(model)
    return tokenizer, model


def compute_metrics(eval_preds: EvalPrediction, tokenizer: PreTrainedTokenizer):
    batched_pred_ids, batched_label_ids = eval_preds

    metrics_dct = {'rouge-1': [], 'rouge-2': [], 'rouge-l': [], 'bleu-4': []}
    for pred_ids, label_ids in zip(batched_pred_ids, batched_label_ids):
        pred_txt = tokenizer.decode(pred_ids).strip()
        label_txt = tokenizer.decode(label_ids).strip()
        pred_tokens = list(jieba.cut(pred_txt))
        label_tokens = list(jieba.cut(label_txt))
        rouge = Rouge()
        scores = rouge.get_scores(' '.join(pred_tokens), ' '.join(label_tokens))
        for k, v in scores[0].items():
            metrics_dct[k].append(round(v['f'] * 100, 4))
        metrics_dct['bleu-4'].append(
            sentence_bleu(
                [label_tokens],
                pred_tokens,
                smoothing_function=SmoothingFunction().method3,
            )
        )
    return {k: np.mean(v) for k, v in metrics_dct.items()}


class TCMDataProcessor:
    """中医数据处理器"""
    
    def __init__(self):
        # 中医属性类别
        self.nature_classes = ["寒", "热", "温", "凉", "平"]
        self.flavor_classes = ["酸", "苦", "甘", "辛", "咸"]
        self.meridian_list = [
            "肺经", "大肠经", "胃经", "脾经", "心经", "小肠经",
            "膀胱经", "肾经", "心包经", "三焦经", "胆经", "肝经"
        ]
        
    def process_tcm_data(self, example: dict) -> dict:
        """处理单条TCM数据
        
        Args:
            example: 包含原始数据的字典
            
        Returns:
            处理后的数据字典
        """
        processed = {}
        
        # 处理症状描述
        if "symptoms" in example:
            processed["input_text"] = f"症状：{example['symptoms']}"
            
        # 处理中医诊断
        if "diagnosis" in example:
            processed["tcm_labels"] = self._process_diagnosis(example["diagnosis"])
            
        # 处理方剂信息
        if "prescription" in example:
            processed["prescription"] = self._process_prescription(example["prescription"])
            
        # 处理中药属性
        if "herbs" in example:
            processed["tcm_attr_labels"] = self._process_herb_attributes(example["herbs"])
            
        return processed
        
    def _process_diagnosis(self, diagnosis: str) -> dict:
        """处理中医诊断信息"""
        # 提取证型信息
        patterns = []
        if "证" in diagnosis:
            patterns = [p for p in ["虚证", "实证", "寒证", "热证", "表证", "里证"] if p in diagnosis]
            
        # 提取病位信息
        locations = []
        for meridian in self.meridian_list:
            if meridian in diagnosis:
                locations.append(meridian)
                
        return {
            "patterns": patterns,
            "locations": locations
        }
        
    def _process_prescription(self, prescription: str) -> dict:
        """处理方剂信息"""
        # 解析方剂组成
        herbs = []
        dosages = []
        
        # 假设格式为："药名1 10g，药名2 15g"
        for item in prescription.split("，"):
            if not item.strip():
                continue
            parts = item.strip().split()
            if len(parts) >= 2:
                herbs.append(parts[0])
                dosage = parts[1].replace("g", "").replace("克", "")
                try:
                    dosages.append(float(dosage))
                except ValueError:
                    dosages.append(0.0)
                    
        return {
            "herbs": herbs,
            "dosages": dosages
        }
        
    def _process_herb_attributes(self, herbs: list) -> dict:
        """处理中药属性"""
        # 初始化属性标签
        nature = torch.zeros(len(self.nature_classes))
        flavor = torch.zeros(len(self.flavor_classes))
        meridian = torch.zeros(len(self.meridian_list))
        
        # 处理每味药的属性
        for herb in herbs:
            # 性质
            if "性质" in herb:
                for i, n in enumerate(self.nature_classes):
                    if n in herb["性质"]:
                        nature[i] = 1
                        
            # 味道
            if "味道" in herb:
                for i, f in enumerate(self.flavor_classes):
                    if f in herb["味道"]:
                        flavor[i] = 1
                        
            # 归经
            if "归经" in herb:
                for i, m in enumerate(self.meridian_list):
                    if m in herb["归经"]:
                        meridian[i] = 1
                        
        return {
            "nature": nature,
            "flavor": flavor,
            "meridian": meridian
        }

def process_tcm_batch(
        batch: Mapping[str, Sequence],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int,
        max_output_length: int,
) -> dict[str, list]:
    """处理TCM数据批次"""
    processor = TCMDataProcessor()
    
    # 处理每条数据
    processed_examples = []
    for example in batch:
        processed = processor.process_tcm_data(example)
        processed_examples.append(processed)
        
    # 转换为模型输入格式
    model_inputs = {
        "input_ids": [],
        "labels": [],
        "tcm_labels": [],
        "tcm_attr_labels": []
    }
    
    for example in processed_examples:
        # 编码输入文本
        inputs = tokenizer(
            example["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        model_inputs["input_ids"].append(inputs["input_ids"][0])
        
        # 添加TCM标签
        if "tcm_labels" in example:
            model_inputs["tcm_labels"].append(example["tcm_labels"])
            
        # 添加属性标签
        if "tcm_attr_labels" in example:
            model_inputs["tcm_attr_labels"].append(example["tcm_attr_labels"])
            
    return model_inputs


@app.command()
def main(
        data_dir: Annotated[str, typer.Argument(help='')],
        model_dir: Annotated[
            str,
            typer.Argument(
                help='A string that specifies the model id of a pretrained model configuration hosted on huggingface.co, or a path to a directory containing a model configuration file.'
            ),
        ],
        config_file: Annotated[str, typer.Argument(help='')],
        auto_resume_from_checkpoint: str = typer.Argument(
            default='',
            help='If entered as yes, automatically use the latest save checkpoint. If it is a numerical example 12 15, use the corresponding save checkpoint. If the input is no, restart training'
        ),

):
    ft_config = FinetuningConfig.from_file(config_file)
    tokenizer, model = load_tokenizer_and_model(model_dir, peft_config=ft_config.peft_config)
    data_manager = DataManager(data_dir, ft_config.data_config)

    train_dataset = data_manager.get_dataset(
        Split.TRAIN,
        functools.partial(
            process_batch,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    print('train_dataset:', train_dataset)
    val_dataset = data_manager.get_dataset(
        Split.VALIDATION,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if val_dataset is not None:
        print('val_dataset:', val_dataset)
    test_dataset = data_manager.get_dataset(
        Split.TEST,
        functools.partial(
            process_batch_eval,
            tokenizer=tokenizer,
            max_input_length=ft_config.max_input_length,
            max_output_length=ft_config.max_output_length,
        ),
        batched=True,
    )
    if test_dataset is not None:
        print('test_dataset:', test_dataset)

    # checks encoded dataset
    _sanity_check(
        train_dataset[0]["input_ids"], train_dataset[0]["labels"], tokenizer
    )

    # turn model to fp32
    _prepare_model_for_training(model, ft_config.training_args.use_cpu)

    ft_config.training_args.generation_config.pad_token_id = (
        tokenizer.pad_token_id
    )
    ft_config.training_args.generation_config.eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command('<|user|>'),
        tokenizer.get_command('<|observation|>'),
    ]
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    use_tokenizer = True
    if ft_config.peft_config is not None:
        use_tokenizer = False if ft_config.peft_config.peft_type == "LORA" else True

    trainer = Seq2SeqTrainer(
        model=model,
        args=ft_config.training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding='longest',
            return_tensors='pt',
        ),
        train_dataset=train_dataset,
        eval_dataset=val_dataset.select(list(range(50))),
        tokenizer=tokenizer if use_tokenizer else None,  # LORA does not need tokenizer
        compute_metrics=functools.partial(compute_metrics, tokenizer=tokenizer),
    )

    if auto_resume_from_checkpoint.upper() == "" or auto_resume_from_checkpoint is None:
        trainer.train()
    else:
        def do_rf_checkpoint(sn):
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            checkpoint_directory = os.path.join(output_dir, "checkpoint-" + sn)
            print("resume checkpoint from  checkpoint-" + sn)
            trainer.train(resume_from_checkpoint=checkpoint_directory)

        output_dir = ft_config.training_args.output_dir

        # resume from latest checkpoint
        if auto_resume_from_checkpoint.upper() == "YES":
            dirlist = os.listdir(output_dir)
            checkpoint_sn = 0
            # get latest checkpoint
            for checkpoint_str in dirlist:
                if checkpoint_str.find("eckpoint") > 0 and checkpoint_str.find("tmp") == -1:
                    checkpoint = int(checkpoint_str.replace("checkpoint-", ""))
                    if checkpoint > checkpoint_sn:
                        checkpoint_sn = checkpoint
            if checkpoint_sn > 0:
                do_rf_checkpoint(str(checkpoint_sn))
            else:
                trainer.train()
        else:
            # resume from specific checkpoint
            if auto_resume_from_checkpoint.isdigit() and int(auto_resume_from_checkpoint) > 0:
                do_rf_checkpoint(auto_resume_from_checkpoint)
            else:
                print(auto_resume_from_checkpoint,
                      "The specified checkpoint sn(" + auto_resume_from_checkpoint + ") has not been saved. Please search for the correct chkeckpoint in the model output directory")

    # test stage
    if test_dataset is not None:
        trainer.predict(test_dataset)


if __name__ == '__main__':
    app()
