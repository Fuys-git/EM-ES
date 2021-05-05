# EM-ES
以entity markers-entity start为关系学习方法的实验代码

few-shot任务

数据集FewRel  评价指标accuracy

1-shot和5-shot数据的收集和评价不太一样

5shot需要对模型加一个线性层来对输出降维

trainer.py

bert-base  ..model.BERT.modeling_bert import BertModel

bert-large  ..transformer.model.bert import BertModel

roberta  ..gaojie_transformers.model.roberta import RobertaModel

roberta-large  ..transformer.model.roberta import RobertaModel



