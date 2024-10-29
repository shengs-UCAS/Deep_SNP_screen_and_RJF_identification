
# Deep_SNP_Screen_and_RJF_identification （DeepSNP&RJF-ID）


DeepSNP&RJF-ID is a tool for molecular identification of Red Jungle Fowl (RJF). Sample the genotype of the chicken to be identified on the specified SNPs (~700) and input it into the AI ​​model in the tool to get the identification result of whether it is a red jungle fowl.

In addition to providing the trained identification model, the tool also provides the entire training process code. The whole process includes

1. Training data preparation: parse the sample VCF file and join label information.
2. SNP fine screening: Use computer perturbation experiments to screen SNPs with high information content and high contribution to identification.
3. Identification model training: train a new model from scratch
4. Qualification model testing: Use the interface of the qualification model.

In addition to the molecular identification of red junglefowl, the tool can also be used for the molecular identification of other species.


<p align="center">

<img src="https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification/blob/main/docs/main_1.png" width="900" height="280">

</p>


## Installation

First, copy the github folder and go into  folder:

```
git clone https://github.com/icefire080/Deep_SNP_screen_and_RJF_identification.git

cd Deep_SNP_screen_and_RJF_identification
```

Next, configure an Conda environment 

```bash

conda env create --file environment.yml

```

After setting up the environment, activate it:

```bash

conda activate smsnpi

```

Now you are ready to run your own analysis 

  

## Quick Usage

### Use the trained model to identify samples

Step 1. Prepare the VCF file of the sample as required, which must contain the following SNPs

```
less trained_model/final_model_700snp.snp_tokens
```

Step 2, use script for identification and scoring

```
sh scripts/reproduce_result_analysis.sh
```

### Train a model from scratch

Step 1, prepare training data and validation data

```
sh scripts/reproduce_sample.sh
```

Step 2: Start SNP fine screening based on perturbation experiments

```
sh scripts/preproduce_snp_choose.sh
```

Step 3, train identification model and data verification

```
sh scripts/reproduce_model_train_test.sh
```

## Contributors

DeepSNP&RJF-ID was developed primarily by Liu Bei with contributions and suggestions by Cai Zhengfei. Supervision for the project was provided by MS.Peng


## Contact

If you enconter a problem when using the software, you can

1. post an issue 
