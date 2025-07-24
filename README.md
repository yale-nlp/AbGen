## AbGen
[**🤗 Dataset**](https://huggingface.co/datasets/yale-nlp/AbGen) | [**📖 arXiv**]() | [**GitHub**](https://github.com/yale-nlp/AbGen)

The data and code for the ACL 2025 paper [AbGen: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research](https://github.com/yale-nlp/AbGen). **AbGen** is designed to evaluate the capabilities of LLMs in designing ablation studies for scientific research. It consists of 1,500 expert-annotated examples derived from 807 NLP papers. In this benchmark, LLMs are tasked with generating detailed ablation study designs for a specified module or process based on the given research context. We also develop **AbGen-Eval**, a meta-evaluation benchmark designed to assess the reliability of commonly used automated evaluation systems in measuring LLM performance on our task.

## AbGen Dataset
All the data examples were divided into two subsets: *validation* and *test*.

- **validation**: 500 examples used for model development, validation, or for those with limited computing resources.
- **test**: 1000 examples for standard evaluation. 
- **AbGen-Eval (human evaluation)**: 1800 examples with human evaluation scores in the aspect of faithfulness, soundness, and importance.

You can download this dataset by the following command:

```python
from datasets import load_dataset

dataset = load_dataset("yale-nlp/AbGen")

# print the first example on the validation set
print(dataset["validation"][0])

# print the first example on the test set
print(dataset["test"][0])

print(dataset["human_evaluation"][0])
```

The dataset is provided in json format and contains the following attributes:

```
{
    "example_id": [string] The example id,
    "arxiv_id": [string] The arxiv id of the paper,
    "title": [string] The title of the paper, 
    "research_background": [string] , which is restructured from the introduction and related work sections, describing the paper's motivation, research problem, and relevant prior work,    "method": [string] which is restructured from the methodology sections, This section describes the proposed method or model, including key components and innovations,
    "main_experiment": [dict],
        "experiment_setup": [string] The setup of the main experiment, which includes the dataset, evaluation metrics, and experimental settings,
        "results": [string] The results of the main experiment, which includes the performance
    "ablation_study": [dict],
        "module_name": [string] The name of the module or process to be ablated,
        "research_objective": [string] a one- or two-sentence description of the research problem and the goal of the ablation study,
        "experiment_setup": [string] a detailed account of the experimental setup, including the experimental groups, datasets, procedures, and the evaluation tools and metrics used,
        "results": [string] an analysis of the outcomes, where annotators summarize the key findings and their implications
}
```

## Experiments
### Environment Setup
The code is tested on the following environment:
- python 3.11.5
- CUDA 12.4, PyTorch 2.1.1
- run `pip install -r requirements.txt` to install all the required packages

### LLM Inference
- `inference/api_inference.py` for running proprietary models or any OpenAI-compatible APIs.
- `inference/vllm_inference.py` for running open-sourced LLMs (e.g., Llama, Mistral, QWen) that are reported in the paper and supported by the [vLLM](https://github.com/vllm-project/vllm) framework with GPUs.
- `inference/evaluation.py` for running the evaluation of the generated ablation study designs using GPT-4.1-mini as the evaluator.

### Model Output
The model outputs and evaluation results on both the validation and test sets can be found at the `model_outputs` directory.

### Meta Evaluation
- `meta_evaluation/inference` include files for running the meta-evaluation of the generated ablation study designs using API-based or vLLM-based inference.
- `meta_evaluation/score_extraction.py` for using rule-based methods to extract scores from the generated ablation study designs (if structured responses is not used).
- `meta_evaluation/correlation_calculation.py` for calculating the system- and instance-level correlation between the LLM-based evaluation scores and human evaluation scores.

## Contact
For any issues or questions, kindly email us at: Yilun Zhao (yilun.zhao@yale.edu).

## Citation

If you use the **AbGen** dataset in your work, please kindly cite the paper:
```
@inproceedings{zhao-etal-2025-abgen,
    title = "{A}b{G}en: Evaluating Large Language Models in Ablation Study Design and Evaluation for Scientific Research",
    author = "Zhao, Yilun  and
      Chen, Weiyuan  and
      Xu, Zhijian  and
      Patwardhan, Manasi  and
      Wang, Chengye  and
      Liu, Yixin  and
      Vig, Lovekesh  and
      Cohan, Arman",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.611/",
    pages = "12479--12491",
    ISBN = "979-8-89176-251-0",
    abstract = "We introduce AbGen, the first benchmark designed to evaluate the capabilities of LLMs in designing ablation studies for scientific research. AbGen consists of 2,000 expert-annotated examples derived from 677 NLP papers. In this benchmark, LLMs are tasked with generating detailed ablation study designs for a specified module or process based on the given research context. Our evaluation of leading LLMs, such as GPT-4o and Llama-3.1, highlights a significant performance gap between these models and human experts in terms of the importance, faithfulness, and soundness of the ablation study designs. Moreover, we demonstrate that current automated evaluation methods are not reliable for our task, as they show a significant discrepancy when compared to human assessment. To better investigate this, we develop AbGen-Eval, a meta-evaluation benchmark designed to assess the reliability of commonly used automated evaluation systems in measuring LLM performance on our task. We investigate various LLM-based evaluation methods on AbGen-Eval, providing insights for future research on developing more effective and reliable LLM-based evaluation systems for complex scientific tasks."
}
```
