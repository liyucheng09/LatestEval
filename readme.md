<p align="center">
    <img src="https://github.com/liyucheng09/LatestEval/blob/master/figs/logo.png" alt="Logo of Selective Context" width="auto" height="160" />
</p>

# "Uncheatable" LLMs Evaluation - LatestEval

Humans receive new test questions every exam, but LLMs? They've been evaluated with the same benchmarks for too long. Why not assess LLMs with fresh test just like we test our students? In this project, we introduce LatestEval, which automatically constructs language model benchmarks using the latest materials (e.g., arXiv, BBC, Wikipedia, etc.) to prevent "cheating" and data contamination.

**News!!**

- **15 Dec, 2023** - This project was accpeted by the main track of **AAAI 2024** :partying_face:! Check out the paper here: :point_right: [Dynamic Test Construction with Latest Materials](https://arxiv.org/abs/2312.12343).

# Key Features

1. We maintain a QA benchmark that updates every half month using the latest online resources (created in the past half month). This approach aims to avoid 1) LLMs being trained on the test set (cheating); and 2) the unintentional inclusion of test questions in the training dataset (data contamination).
2. We analyzed real Human-AI conversations to ensure the automated benchmark aligns well with real-life applications (see [paper](https://arxiv.org/abs/2312.12343) for more detail).


# The Benchmark

Access the latest benchmark dorectly at [Huggingface Hub](https://huggingface.co/LatestEval)!

- Latest benchmark of GitHub: [HF Hub](https://huggingface.co/datasets/LatestEval/github-latest)
- Latest benchmark of arXiv: [HF Hub](https://huggingface.co/datasets/LatestEval/arxiv-latest)
- Latest benchmark of BBC: [HF Hub](https://huggingface.co/datasets/LatestEval/bbc-latest)
- The Full benchmark with all sources: [HF Hub](https://huggingface.co/datasets/LatestEval/full-latest)

The benchmarks are created with latest materials, find these raw materials/documents at [Huggingface Hub](https://huggingface.co/RealTimeData)

# Evaluate your LLM on LatestEval

We will add LatestEval to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) and [OpenCompass](https://github.com/open-compass/opencompass). Stay tuned.

# Create benchmarks with your own data

1. Put your documents as `.txt` files under `./<your_doc_path>`.
2. Set your OpenAI key:

```
export OPENAI_API_KEY=<Your OpenAI key>
```

3. Simply run:

```
python data_processor.py --source customized --file_path <your_path> --num_docs 100
```

If you want to reproduce LatestEval on arXiv, BBC, GitHub:

```
python data_processor.py --source arxiv --num_docs 100
```

# Issue

Open an issue if you have any problems or want to discuss.

# Citation

If you find this project useful, consider cite this project:

```
@misc{li2023avoiding,
      title={Avoiding Data Contamination in Language Model Evaluation: Dynamic Test Construction with Latest Materials}, 
      author={Yucheng Li and Frank Guerin and Chenghua Lin},
      year={2023},
      eprint={2312.12343},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```