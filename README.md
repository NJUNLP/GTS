# GTS
[Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction](https://arxiv.org/pdf/2010.04640.pdf). Zhen Wu, Chengcan Ying, Fei Zhao, Zhifang Fan, Xinyu Dai, Rui Xia. In Findings of EMNLP, 2020.

## Data
[[Data](https://github.com/NJUNLP/GTS/tree/main/data)]   [[Pre-trained Model](https://drive.google.com/drive/folders/15HZun7FeObpNaJF1gwrJxn2H6e28LPZY?usp=sharing)(from huggingface)]. Data format descriptions are [here](https://github.com/NJUNLP/GTS/blob/main/data/datareadme.md).

**❗Note: our opinion triplet datasets are completely from alignments of our previous work [TOWE](https://www.aclweb.org/anthology/N19-1259/) datasets and the original SemEval [2014](https://www.aclweb.org/anthology/S14-2004/), [2015](https://www.aclweb.org/anthology/S15-2082/), [2016](https://www.aclweb.org/anthology/S16-1002/) datasets, which are different from others.**

## Requirements
See requirement.txt or Pipfile for details
* pytorch==1.7.1
* transformers==3.4.0
* python=3.6

## Usage
- ### Training
For example, you can use the folowing command to fine-tune Bert on the OPE task (the pre-trained Bert model is saved in the folder "pretrained/"):
```
python main.py --task pair --mode train --dataset res14
```
The best model will be saved in the folder "savemodel/".

- ### Testing
For example, you can use the folowing command to test Bert on the OPE task:
```
python main.py --task pair --mode test --dataset res14
```

**Note**: In our pre-experiments, a smaller batch size and learning rate can achieve better performance on certain datasets, while we use a general setting in our paper to save time instead of adopting grid search.

## Citation
If you used the datasets or code, please cite our paper:
```bibtex
@inproceedings{wu-etal-2020-grid,
    title = "Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction",
    author = "Wu, Zhen  and
      Ying, Chengcan  and
      Zhao, Fei  and
      Fan, Zhifang  and
      Dai, Xinyu  and
      Xia, Rui",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.234",
    doi = "10.18653/v1/2020.findings-emnlp.234",
    pages = "2576--2585",
}
```
```bibtex
@inproceedings{fan-etal-2019-target,
    title = "Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling",
    author = "Fan, Zhifang  and
      Wu, Zhen  and
      Dai, Xin-Yu  and
      Huang, Shujian  and
      Chen, Jiajun",
    booktitle = "Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1259",
    doi = "10.18653/v1/N19-1259",
    pages = "2509--2518",
}
```

## Reference
[1]. Zhen Wu, Chengcan Ying, Fei Zhao, Zhifang Fan, Xinyu Dai, Rui Xia. [Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction](https://arxiv.org/pdf/2010.04640.pdf). In Findings of EMNLP, 2020.

[2]. Zhifang Fan, Zhen Wu, Xin-Yu Dai, Shujian Huang, Jiajun Chen. [Target-oriented Opinion Words Extraction with Target-fused Neural Sequence Labeling](https://www.aclweb.org/anthology/N19-1259.pdf). In Proceedings of NAACL, 2019.
