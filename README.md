# GTS
[Grid Tagging Scheme for Aspect-oriented Fine-grained Opinion Extraction](https://arxiv.org/pdf/2010.04640.pdf). Zhen Wu, Chengcan Ying, Fei Zhao, Zhifang Fan, Xinyu Dai, Rui Xia. In Findings of EMNLP, 2020.

## Data
**❗Note: Our opinion triplet datasets are completely from alignments of our previous work [TOWE](https://www.aclweb.org/anthology/N19-1259/) datasets and the original SemEval [2014](https://www.aclweb.org/anthology/S14-2004/), [2015](https://www.aclweb.org/anthology/S15-2082/), [2016](https://www.aclweb.org/anthology/S16-1002/) datasets. GTS datasets contain the cases of one aspect term corresponding to multiple opinion terms and one opinion term corresponding to multiple aspect terms.**

[[Data](https://github.com/NJUNLP/GTS/tree/main/data)]   [[Pre-trained Model](https://drive.google.com/drive/folders/15HZun7FeObpNaJF1gwrJxn2H6e28LPZY?usp=sharing)(from huggingface)]. Data format descriptions are [here](https://github.com/NJUNLP/GTS/blob/main/data/datareadme.md).

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

# Results
GTS performance on [opinion pair extraction (OPE) datasets](https://github.com/NJUNLP/GTS/tree/main/data):
<table>
	<tr>
	    <th rowspan="2">Methods</th>
	    <th colspan="3">14res</th>
	    <th colspan="3">14lap</th>
      <th colspan="3">15res</th>
      <th colspan="3">16res</th>  
	</tr >
  <tr >
	    <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
	</tr>
	<tr >
	    <td>GTS-CNN</td>
	    <td>74.13</td>
	    <td>69.49</td>
      <td>71.74</td>
      <td>68.33</td>
      <td>55.04</td>
      <td>60.97</td>
      <td>66.81</td>
      <td>61.34</td>
      <td>63.96</td>
      <td>70.48</td>
      <td>72.39</td>
      <td>71.42</td>
	</tr>
  <tr >
	    <td>GTS-BiLSTM</td>
	    <td>71.32</td>
	    <td>67.07</td>
      <td>69.13</td>
      <td>61.53</td>
      <td>54.31</td>
      <td>57.69</td>
      <td>67.76</td>
      <td>63.19</td>
      <td>65.39</td>
      <td>70.32</td>
      <td>70.46</td>
      <td>70.39</td>
	</tr>
  <tr >
	    <td>GTS-BERT</td>
	    <td>76.23</td>
	    <td>74.84</td>
      <td>75.53</td>
      <td>66.41</td>
      <td>64.95</td>
      <td>65.67</td>
      <td>66.40</td>
      <td>68.71</td>
      <td>67.53</td>
      <td>71.70</td>
      <td>77.79</td>
      <td>74.62</td>
	</tr>
</table>

GTS performance on [opinion triplet extraction (OTE) datasets](https://github.com/NJUNLP/GTS/tree/main/data):
<table>
	<tr>
	    <th rowspan="2">Methods</th>
	    <th colspan="3">14res</th>
	    <th colspan="3">14lap</th>
      <th colspan="3">15res</th>
      <th colspan="3">16res</th>  
	</tr >
  <tr >
	    <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
	</tr>
	<tr >
	  <td>GTS-CNN</td>
	  <td>70.79</td>
	  <td>61.71</td>
      <td>65.94</td>
      <td>55.93</td>
      <td>47.52</td>
      <td>51.38</td>
      <td>60.09</td>
      <td>53.57</td>
      <td>56.64</td>
      <td>62.63</td>
      <td>66.98</td>
      <td>64.73</td>
	</tr>
  <tr >
	  <td>GTS-BiLSTM</td>
	  <td>67.28</td>
	  <td>61.91</td>
      <td>64.49</td>
      <td>59.42</td>
      <td>45.13</td>
      <td>51.30</td>
      <td>63.26</td>
      <td>50.71</td>
      <td>56.29</td>
      <td>66.07</td>
      <td>65.05</td>
      <td>65.56</td>
	</tr>
  <tr >
	  <td>GTS-BERT</td>
	  <td>70.92</td>
	  <td>69.49</td>
      <td>70.20</td>
      <td>57.52</td>
      <td>51.92</td>
      <td>54.58</td>
      <td>59.29</td>
      <td>58.07</td>
      <td>58.67</td>
      <td>68.58</td>
      <td>66.60</td>
      <td>67.58</td>
	</tr>
</table>

GTS performance on [ASTE-Data-V2 datasets](https://arxiv.org/pdf/2010.02609.pdf):
<table>
	<tr>
	    <th rowspan="2">Methods</th>
	    <th colspan="3">14res</th>
	    <th colspan="3">14lap</th>
      <th colspan="3">15res</th>
      <th colspan="3">16res</th>  
	</tr >
  <tr >
	    <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
      <th>P</th>
	    <th>R</th>
	    <th>F1</th>
	</tr>
	<tr >
	  <td>GTS-BERT</td>
	  <td>68.71</td>
	  <td>67.67</td>
      <td>68.17</td>
      <td>58.54</td>
      <td>50.65</td>
      <td>54.30</td>
      <td>60.69</td>
      <td>60.54</td>
      <td>60.61</td>
      <td>67.39</td>
      <td>66.73</td>
      <td>67.06</td>
	</tr>
</table>
    
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
