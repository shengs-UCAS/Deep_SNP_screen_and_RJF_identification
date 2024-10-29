
# Deep_SNP_Screen_and_RJF_identification （DeepSNP&RJF-ID）


DeepSNP&RJF-ID 是一个对红原鸡进行分子鉴定的工具。采样待鉴定的鸡在指定SNPs（~700）上的基因型，输入到工具内的AI模型即可得到是否是红原鸡的鉴定结果。

工具除了提供训练好的鉴定模型，还提供了整个训练过程代码。整个过程包括

1. 训练数据准备：样品VCF文件解析并拼接标签信息。
2. SNP位点精筛：采用计算机微扰实验，筛选信息量高、对鉴定贡献度高的SNPs。
3. 鉴定模型训练：重头训练新的模型
4. 鉴定模型测试：使用鉴定模型的接口。

工具除了用于红原鸡分子鉴定外，也可以用于其他物种的分子鉴定。


方法的整体示意图如下：


<p align="center">

<img src="https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification/blob/main/docs/main_1.png" width="900" height="280">

</p>


## 安装

clone source code

```bash

git clone https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification.git

cd Deep_SNP_screen_and_RJF_identification

```

配置Conda环境

```bash

conda env create --file environment.yml

```

激活环境

```bash

conda activate smsnpi

```

## 快速使用

### 使用训练好的模型对样品进行鉴定

步骤1， 按要求准备样品的VCF文件，必须包含以下SNPs

```
less trained_model/final_model_700snp.snp_tokens
```

步骤2，使用脚本进行鉴定打分

```
sh scripts/reproduce_result_analysis.sh
```

### 从头开始训练模型

步骤1，准备训练数据和验证数据

```
sh scripts/reproduce_sample.sh
```

步骤2，开始基于微扰实验的SNP精筛

```bash
sh scripts/preproduce_snp_choose.sh
```

步骤3，训练鉴定模型和数据验证

```bash
sh scripts/reproduce_model_train_test.sh
```


## 贡献者

DeepSNP&RJF-ID 主要有刘贝进行开发，蔡正飞参与开发与建议。彭旻晟作为项目负责人。

was developed primarily by Zeyang Shen, with contributions and suggestions by Marten Hoeksema and Zhengyu Ouyang. Supervision for the project was provided by Christopher Glass and Christopher Benner.


## 联系

如果你遇到问题，可以在Issue里提交
