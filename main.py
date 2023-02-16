from transformers import pipeline
import hebrew_tokenizer as ht

# how to use?
# NER = pipeline(
#     "token-classification",
#     model="avichr/heBERT_NER",
#     preprocessing="avichr/heBERT_NER",
# )

with open('./verdict/1.txt', 'r',encoding='utf-8') as file:
    lines = file.readlines()
    # for line in lines:
    #     print(line)
    #     # print(NER(line))