# MultiTaskLearn

多任务学习，赛题的地址：https://tianchi.aliyun.com/competition/entrance/531841/information

数据说明
OCNLI：是第一个非翻译的、使用原生汉语的大型中文自然语言推理数据集；
OCEMOTION：是包含7个分类的细粒度情感性分析数据集；
TNEWS：来源于今日头条的新闻版块，共包含15个类别的新闻；

数据集的下载地址：
| 名称                    | 大小     |                                                         Link |
| :---------------------- | :------- | -----------------------------------------------------------: |
| OCEMOTION_a.csv         | 200.77KB | https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/OCEMOTION_a.csv |
| OCNLI_a.csv             | 165.17KB | https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/OCNLI_a.csv |
| TNEWS_a.csv             | 98.39KB  | https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/TNEWS_a.csv |
| OCNLI_train1128.csv     | 5.78MB   | http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/OCNLI_train1128.csv |
| TNEWS_train1128.csv     | 4.38MB   | http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/TNEWS_train1128.csv |
| OCEMOTION_train1128.csv | 4.96MB   | http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531841/OCEMOTION_train1128.csv |


文件目录结构：
datasets -- 数据集
logs  -- 日志文件
model  --预训练模型 
bert_model.py  --模型
generate_data.py  --预处理数据
joint_dataset.py -- 生成训练batch数据
train.py  -- 训练
utils.py  --工具
infer.py --推理