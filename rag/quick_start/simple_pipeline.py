import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--retriever_path", type=str)
args = parser.parse_args()

config_dict = {
    "data_dir": "dataset/",
    "index_path": "indexes/e5_Flat.index",
    "corpus_path": "indexes/general_knowledge.jsonl",
    "model2path": {"bge":"F:/work/FlashRAG/model/e5-base-v2",  "Qwen1.5-7B-Chat": "F:/work/FlashRAG/model/Qwen1.5-7B-Chat"},
    "generator_model": "Qwen1.5-7B-Chat",
    "retrieval_method": "bge",
    "metrics": ["em", "f1", "acc","precision","recall"],
    "retrieval_topk": 5,
    "save_intermediate_data": True,
    "generator_max_input_len": 512,
    "max_tokens": 16,
}

config = Config(config_dict=config_dict)

all_split = get_dataset(config)
test_data = all_split["test"]
print("test_data:",test_data)
prompt_templete = PromptTemplate(
    config,
    system_prompt = "请根据提供的文本进行回答，不要输出与文本无关的内容\n文本内容如下：\n\n{reference}",
    user_prompt = "问题: {question}\n答案:"
)
pipeline = SequentialPipeline(config,prompt_template=prompt_templete)

output_dataset = pipeline.run(test_data, do_eval=True)
print("---generation output---")
print(output_dataset.pred)
