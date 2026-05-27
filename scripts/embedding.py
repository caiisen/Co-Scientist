import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
	base_url="https://ai.gitee.com/v1",
	api_key="9KYTDGH2U7UBN3L23PZRDWXSWSPYTOHMTMHQQID4",
	default_headers={"X-Failover-Enabled":"true"},
)

sentences = [
    # 要生成嵌入向量的输入文本
    "Today is a sunny day and I will get some ice cream.",
    "Tomorrow looks rainy, so I will stay at home."
]

response = client.embeddings.create(
	input=sentences,
	model="Qwen3-Embedding-8B", # 指定使用的重排序模型
)

print("Call successful!")
print("Type of response object:", type(response))
print("Number of returned vectors:", len(response.data))

for idx, emb in enumerate(response.data):
    print(f"\nText entry {idx+1}:{sentences[idx]}")
    print(f"Dimensionality of the vector:{len(emb.embedding)}")
    print(f"First 10 values of the vector:{emb.embedding[:10]}") # 仅打印前10个值以确认调用成功