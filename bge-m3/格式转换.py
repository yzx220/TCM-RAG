import json

# 输入JSONL文件路径
input_file = "F:\实验\\all_data\\all_dev_embedding_data.jsonl"
# 输出JSONL文件路径
output_file = "F:\实验\\flashrag_data\\test.jsonl"

# 读取输入文件并转换为目标格式
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "a", encoding="utf-8") as outfile:
    for idx, line in enumerate(infile):
        # 解析JSONL文件中的每一行
        data = json.loads(line.strip())
        query = data.get("query", "")
        positive = data.get("positive", "")

        # 构建目标格式
        output_data = {
            "id": f"test_{idx}",  # 生成唯一ID
            "question": query,  # 使用query作为question
            "golden_answers": [positive]  # 将positive作为golden_answers的值（转换为列表）
        }

        # 将转换后的数据写入输出文件
        outfile.write(json.dumps(output_data, ensure_ascii=False) + "\n")

print(f"转换完成，结果已追加到 {output_file}")