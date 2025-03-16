# Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive Regression

This is the official implementation of our paper [Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive Regression](https://arxiv.org/pdf/2501.03674 ).

## Introduction

Action Quality Assessment (AQA), which aims at automatic and fair evaluation of athletic performance, has gained increasing attention in recent years. However, athletes are often in rapid movement and the corresponding visual appearance variances are subtle, making it challenging to capture fine-grained pose differences and leading to poor estimation performance. Furthermore, most common AQA tasks, such as diving in sports, are usually divided into multiple sub-actions, each of which contains different durations. However, existing methods focus on segmenting the video into fixed frames, which disrupts the temporal continuity of sub-actions resulting in unavoidable prediction errors. To address these challenges, we propose a novel action quality assessment method through hierarchically pose-guided multi-stage contrastive regression. 

Firstly, we introduce a multi-scale dynamic visual-skeleton encoder to capture fine-grained spatio-temporal visual and skeletal features. Then, a procedure segmentation network is introduced to separate different sub-actions and obtain segmented features. Afterwards, the segmented visual and skeletal features are both fed into a multi-modal fusion module as physics structural priors, to guide the model in learning refined activity similarities and variances. Finally, a multi-stage contrastive learning regression approach is employed to learn discriminative representations and output prediction results. In addition, we introduce a newly-annotated FineDiving-Pose Dataset to improve the current low-quality human pose labels.

### **Extended Work**
Compared to the preliminary version, we leverage an additional skeletal modality to obtain hierarchical human pose features. Given the limitations in existing datasets characterized by the poor quality or absence of skeletal labels, we also present a newly-annotated FineDiving-Pose Dataset with refined pose labels, which are collected through a combination of manual annotation and automatic generation to further boost the related field.Furthermore, we propose a multimodal fusion module to integrate visual features and skeletal features and add a static branch to capture human static features. 
## Table of Contents

1. [News](#news)
2. [Data Preparation](#data-preparation)
3. [Train and Eval](#train-and-eval)
4. [Citation](#citation)

## News

- **[08 Jan, 2025]** We have released the Arxiv version of the paper. Code/Models are coming soon. Please stay tuned!
- **[16 Mar, 2025]** We have uploaded the training code and the human-annotated pose data!

## Data Preparation

a. We extracted and processed data from the [FineDiving dataset](https://github.com/xujinglin/FineDiving).

b. We expanded the human-annotated pose data and the automatically annotated pose data.

c. We provide some pose data in the [examples](examples/annotations) for presentation.

d. You can download the human-annotated pose dataset [here](https://pan.baidu.com/s/1Ozhd0c3H-KSqcLdno6WH2Q?pwd=9kq2)


**TODO:** The complete dataset and code will be available once the paper is accepted. Stay tune！
## Train and Eval

The training code is now available!

## Citation

If you find this project useful in your research, please consider citing:

> ```
> @article{qi2025action,
> title={Action Quality Assessment via Hierarchical Pose-guided Multi-stage Contrastive Regression},
> author={Qi, Mengshi and Ye, Hao and Peng, Jiaxuan and Ma, Huadong},
> journal={arXiv preprint arXiv:2501.03674},
> year={2025}
> }
> 