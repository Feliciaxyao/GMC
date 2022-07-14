# GMC_NExT-QA 


## Dataset
NExT-QA is a VideoQA benchmark targeting the explanation of video contents. It challenges QA models to reason about the causal and temporal actions and understand the rich object interactions in daily activities. For more details, please refer to the [link](https://doc-doc.github.io/docs/nextqa.html) page.

## Todo
1. [x] [Relation annotations](https://drive.google.com/file/d/1RW8ck39n-yScGrOZWJ7gBr1jH3Iy4SSl/view?usp=sharing) are available.
2. [x] <s>Open evaluation server</s> and release [test data](https://drive.google.com/file/d/1_MEqDeQHc8Y8Uw7eW58HVuZy2iyThILQ/view?usp=sharing).
3. [x] Release [spatial feature](https://drive.google.com/file/d/1yJ30T1oAjJ8cO3nHQID0EmIm-yQHdYkK/view?usp=sharing) (valid for 2 weeks).
4. [ ] Release RoI feature.
5. [ ] Release BERT finetune code for VQA.

## Environment

Anaconda 4.8.4, python 3.6.8, pytorch 1.6 and cuda 10.2. For other libs, please refer to the file requirements.txt.

## Install
Please create an env for this project using anaconda (should install [anaconda](https://docs.anaconda.com/anaconda/install/linux/) first)
```
>conda create -n videoqa python==3.6.8
>conda activate videoqa
>git clone https://github.com/doc-doc/NExT-QA.git
>pip install -r requirements.txt #may take some time to install
```

## Data Preparation
Please download the pre-computed features and QA annotations from [here](https://drive.google.com/drive/folders/1gKRR2es8-gRTyP25CvrrVtV6aN5UxttF?usp=sharing). There are 4 zip files: 
- ```['vid_feat.zip']```: Appearance and motion feature for video representation. (With code provided by [HCRN](https://github.com/thaolmk54/hcrn-videoqa)).
- ```['qas_bert.zip']```: Finetuned BERT feature for QA-pair representation. (Based on [pytorch-pretrained-BERT](https://github.com/LuoweiZhou/pytorch-pretrained-BERT/)).
- ```['nextqa.zip']```: Annotations of QAs and GloVe Embeddings. 
- ```['models.zip']```: HGA model. 

After downloading the data, please create a folder ```['data/feats']``` at the same directory as ```['NExT-QA']```, then unzip the video and QA features into it. You will have directories like ```['data/feats/vid_feat/', 'data/feats/qas_bert/' and 'NExT-QA/']``` in your workspace. Please unzip the files in ```['nextqa.zip']``` into ```['NExT-QA/dataset/nextqa']``` and ```['models.zip']``` into ```['NExT-QA/models/']```. 


## Usage
Once the data is ready, you can easily run the code. First, to test the environment and code, we provide the prediction and model of the SOTA approach (i.e., HGA) on NExT-QA. 
You can get the results reported in the paper by running: 
```
>python eval_mc.py
```
The command above will load the prediction file under ['results/'] and evaluate it. 
You can also obtain the prediction by running: 
```
>./main.sh 0 val #Test the model with GPU id 0
```
The command above will load the model under ['models/'] and generate the prediction file.
If you want to train the model, please run
```
>./main.sh 0 train # Train the model with GPU id 0
```
It will train the model and save to ['models']. (*The results may be slightly different depending on the environments*)
## Results on Val. Set
| Methods                  | Text Rep. | Acc_C | Acc_T | Acc_D | Acc   |
| -------------------------| --------: | ----: | ----: | ----: | ----: |
| [EVQA](https://github.com/doc-doc/NExT-QA/blob/main/networks/VQAModel/EVQA.py)                     | BERT-FT | 42.46 | 46.34 | 45.82 | 44.24 |
| [STVQA](https://github.com/doc-doc/NExT-QA/blob/main/networks/VQAModel/STVQA.py) ([CVPR17](https://openaccess.thecvf.com/content_cvpr_2017/papers/Jang_TGIF-QA_Toward_Spatio-Temporal_CVPR_2017_paper.pdf))  | BERT-FT | 44.76 | 49.26 | 55.86 | 47.94 |
| [CoMem](https://github.com/doc-doc/NExT-QA/blob/main/networks/VQAModel/CoMem.py) ([CVPR18](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1924.pdf))  | BERT-FT | 45.22 | 49.07 | 55.34 | 48.04 |
| [HCRN](https://github.com/thaolmk54/hcrn-videoqa) ([CVPR20](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Hierarchical_Conditional_Relation_Networks_for_Video_Question_Answering_CVPR_2020_paper.pdf))   | [BERT-FT](https://github.com/doc-doc/HCRN-BERT) | 45.91 | 49.26 | 53.67 | 48.20 |
| [HME](https://github.com/doc-doc/NExT-QA/blob/main/networks/VQAModel/HME.py) ([CVPR19](https://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_Heterogeneous_Memory_Enhanced_Multimodal_Attention_Model_for_Video_Question_Answering_CVPR_2019_paper.pdf))    | BERT-FT | 46.18 | 48.20 | 58.30 | 48.72 |
| [HGA](https://github.com/doc-doc/NExT-QA/blob/main/networks/VQAModel/HGA.py) ([AAAI20](https://ojs.aaai.org//index.php/AAAI/article/view/6767))   | BERT-FT | 46.14 | 50.68 | **59.33** | 49.66 |
| [GMC(MC)](https://github.com/Feliciaxyao/GMC)     | BERT-FT | **47.99** | **50.81** | 58.69 | **50.46** |


## Citation
```
@InProceedings{,
    author    = {YAO Xuan, GAO Jun-Yu, XU Chang-Sheng},
    title     = {Video Question Answering Method Based on Self-supervised Graph Neural Network with Contrastive Learning},
    booktitle = {},
    month     = {},
    year      = {},
    pages     = {}
}
```
## Acknowledgement
Our reproduction of the methods is based on the respective official repositories, we thank the authors to release their code. If you use the related part, please cite the corresponding paper commented in the code.
