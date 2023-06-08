# Evaluating Document Processing Capability of LLMs

Welcome to the Document Processing Evaluation Toolkit (DocEval), a resource for assessing the capabilities of Language Learning Models (LLMs) in processing various types of documents. This toolkit is regularly updated, thus ensuring consistent, reliable, and up-to-date performance analysis of LLMs.

## Key Features
1. **Continuous Updates**: Our test set is updated every two weeks to prevent data contamination. This consistent renewal ensures that the documents and queries in our evaluation framework remain fresh, thus eliminating the possibility of baselines becoming pre-optimized on our test instances.

2. **Comprehensive Performance Analysis**: The evaluation framework assesses performance across three major dimensions, with further subdivision into more fine-grained aspects. This multi-dimensional approach provides a holistic view of an LLM's abilities and potential areas for improvement.

## Data Sources

We consider to include the following types of documents: 

1. Arxiv preprints ✅
2. BBC News ✅
3. Company Anouncements
4. Legal Documents
5. Software release note on Github

## Dimensions to eval

### Fine-grained queries

1. Analysis: This might involve breaking down complex arguments or methods described in the document, or identifying the evidence and reasoning that supports a particular conclusion.

```
- Can you break down the method described in this section?
- What are the main pieces of evidence supporting the argument in this paragraph?"
```

2. Interpretation: Questions about the meaning of specific terms, sentences, or passages in the text, or about the implications of a particular argument or piece of data.

```
- What does this phrase/term mean in the context of this document?
- What are the potential implications of the argument made in this section?
```

3. Searching: Specific requests for information contained in the document, such as details about a method's precision, as found in a table, or the mention of a certain topic or keyword.

```
- Where in the document does it mention [specific topic or term]?
- What does the table on page X say about the precision of this method?
```

4. Comparison: Questions that seek to understand the contrast between ideas, methods, events, or data points in the current document and other sources or previous work.

```
- How does the approach described in this paper differ from previous methods?
- How does the author's viewpoint compare to other perspectives on this issue?
```

### Overall understanding

1. Summary related queris: Asking the agent to distill the essential information from a specific part of the document or the entire text.

```
- Could you summarize the main points of this document?
- What is the key message of this section?
```

### Predictive queries

1. Potential Application and Consequence: Questions about the possible real-world applications of a scientific method, the likely effects of a policy change, or the impact of a particular event or decision.

```
- What might be some potential applications of the findings of this research paper?
- What could be the impact of the policy change discussed in this document?
```

2. Hypotheticals ("What if" scenarios): Queries that explore alternative scenarios or outcomes, often requiring the AI to extrapolate from the information in the document.

```
- What might have happened if [alternative scenario] instead?
- How might the situation change if [specific variable or factor] were different?
```

3. Decision Justification: Questions about the reasoning or motivations behind decisions made by individuals or organizations, as described in the document. This might involve both factual information from the text and inferences based on the context or the AI's general knowledge.

```
- Why did the company choose to implement this strategy, according to the document?
- What reasons does the author give for [individual or organization]'s decision?
```

## Example

Here is a snippet from the LLaMA preprint.

```
\section{Approach}
Our training approach is similar to the methods described in previous work~\cite{brown2020gpt3,chowdhery2022palm}, and is inspired by the Chinchilla scaling laws~\cite{hoffmann2022chinchilla}.
We train large transformers on a large quantity of textual data using a standard optimizer.
%

\subsection{Pre-training Data}
Our training dataset is a mixture of several sources, reported in Table~\ref{tab:dataset}, that cover a diverse set of domains.
For the most part, we reuse data sources that have been leveraged to train other LLMs, with the restriction of only using data that is publicly available, and compatible with open sourcing.
This leads to the following mixture of data and the percentage they represent in the training set:

\paragraph{English CommonCrawl [67\%].} 
We preprocess five CommonCrawl dumps, ranging from 2017 to 2020, with the CCNet pipeline~\cite{wenzek-etal-2020-ccnet}.
This process deduplicates the data at the line level, performs language identification with a fastText linear classifier to remove non-English pages and filters low quality content with an n-gram language model.
In addition, we trained a linear model to classify pages used as references in Wikipedia \emph{v.s.} randomly sampled pages, and discarded pages not classified as references.


\paragraph{C4 [15\%].}
During exploratory experiments, we observed that using diverse pre-processed CommonCrawl datasets improves performance.
We thus included the publicly available C4 dataset~\citep{raffel2020exploring} in our data.
The preprocessing of C4 also contains deduplication and language identification steps: the main difference with CCNet is the quality filtering, which mostly relies on heuristics such as presence of punctuation marks or the number of words and sentences in a webpage.

\paragraph{Github [4.5\%].}
We use the public GitHub dataset available on Google BigQuery.
We only kept projects that are distributed under the Apache, BSD and MIT licenses.
Additionally, we filtered low quality files with heuristics based on the line length or proportion of alphanumeric characters, and removed boilerplate, such as headers, with regular expressions.
Finally, we deduplicate the resulting dataset at the file level, with exact matches.

\paragraph{Wikipedia [4.5\%].}
We add Wikipedia dumps from the June-August 2022 period, covering 20 languages, which use either the Latin or  Cyrillic scripts: \texttt{bg}, \texttt{ca}, \texttt{cs}, \texttt{da}, \texttt{de}, \texttt{en}, \texttt{es}, \texttt{fr}, \texttt{hr}, \texttt{hu}, \texttt{it}, \texttt{nl}, \texttt{pl}, \texttt{pt}, \texttt{ro}, \texttt{ru}, \texttt{sl}, \texttt{sr}, \texttt{sv}, \texttt{uk}.
We process the data to remove hyperlinks, comments and other formatting boilerplate.

\begin{table}[t]
  \center
   \setlength{\tabcolsep}{3pt}
  \begin{tabular}{@{}l@{}ccr@{}}
  \toprule
  Dataset &  Sampling prop. & Epochs &  Disk size \\  % 
  \midrule
  CommonCrawl    & 67.0\%  & 1.10 & 3.3 TB \\
  C4             & 15.0\%  & 1.06 & 783 GB \\
  Github         & ~~4.5\% & 0.64 & 328 GB \\
  Wikipedia      & ~~4.5\% & 2.45 & 83 GB \\ 
  Books          & ~~4.5\% & 2.23 & 85 GB \\
  ArXiv          & ~~2.5\% & 1.06 & 92 GB \\
  StackExchange & ~~2.0\%   & 1.03 & 78 GB \\
  \bottomrule
  \end{tabular}
  \caption{\textbf{Pre-training data.} Data mixtures used for pre-training, for each subset we list the sampling proportion, number of epochs performed on the subset when training on 1.4T tokens, and disk size. The pre-training runs on 1T tokens have the same sampling proportion. %
  \label{tab:dataset}
  }
\end{table}

\paragraph{Gutenberg and Books3 [4.5\%].}
We include two book corpora in our training dataset: the Gutenberg Project, which contains books that are in the public domain, and the Books3 section of ThePile~\citep{pile}, a publicly available dataset for training large language models.
We perform deduplication at the book level, removing books with more than 90\% content overlap.

\paragraph{ArXiv [2.5\%].}
We process arXiv Latex files to add scientific data to our dataset. Following \citet{lewkowycz2022solving}, we removed everything before the first section, as well as the bibliography. We also removed the comments from the .tex files, and inline-expanded definitions and macros written by users to increase consistency across papers.

\paragraph{Stack Exchange [2\%].}
We include a dump of Stack Exchange, a website of high quality questions and answers that covers a diverse set of domains, ranging from computer science to chemistry.
We kept the data from the 28 largest websites, removed the HTML tags from text and sorted the answers by score (from highest to lowest).

\begin{table*}[t!]
\center
\begin{tabular}{ccccccc}
\toprule
  params & dimension & $n$ heads & $n$ layers & learning rate & batch size & $n$ tokens \\
\midrule
  6.7B  & 4096 & 32 & 32 & $3.0e^{-4}$ & 4M & 1.0T \\
  13.0B & 5120 & 40 & 40 & $3.0e^{-4}$ & 4M & 1.0T \\
  32.5B & 6656 & 52 & 60 & $1.5e^{-4}$ & 4M & 1.4T \\
  65.2B & 8192 & 64 & 80 & $1.5e^{-4}$ & 4M & 1.4T \\
\bottomrule
\end{tabular}
\caption{
\textbf{Model sizes, architectures, and optimization hyper-parameters.}
\label{tab:architecture}
}
\end{table*}

\paragraph{Tokenizer.}
We tokenize the data with the byte-pair encoding (BPE) algorithm~\citep{sennrich2015neural}, using the implementation from SentencePiece~\citep{kudo2018sentencepiece}.
Notably, we split all numbers into individual digits, and fallback to bytes to decompose unknown UTF-8 characters.

Overall, our entire training dataset contains roughly 1.4T tokens after tokenization.
For most of our training data, each token is used only once during training, with the exception of the Wikipedia and Books domains, over which we perform approximately two epochs.

\subsection{Architecture}

Following recent work on large language models, our network is based on the transformer architecture~\cite{vaswaniAttention2017}.
We leverage various improvements that were subsequently proposed, and used in different models such as PaLM.
Here are the main difference with the original architecture, and where we were found the inspiration for this change (in bracket):

\paragraph{Pre-normalization [GPT3].} To improve the training stability, we normalize the input of each transformer sub-layer, instead of normalizing the output. We use the RMSNorm normalizing function, introduced by \citet{zhang2019root}.

\paragraph{SwiGLU activation function [PaLM].} We replace the ReLU non-linearity by the SwiGLU activation function, introduced by \citet{shazeer2020glu} to improve the performance. We use a dimension of $\frac23 4d$ instead of $4d$ as in PaLM.

\paragraph{Rotary Embeddings [GPTNeo].}\hspace{-3pt}We remove the absolute positional embeddings, and instead, add rotary positional embeddings (RoPE), introduced by \citet{su2021roformer}, at each layer of the network.

The details of the hyper-parameters for our different models are given in Table~\ref{tab:architecture}.
...
```

We have the following queries:

**For details**

1. Analysis:
   - What are the implications of using rotary embeddings instead of absolute positional embeddings in the model architecture?

2. Interpretation:
   - What does the term 'Chinchilla scaling laws' mean in the context of training large language models?

3. Searching:
   - What does Table 2 indicate about the model sizes and optimization hyper-parameters?

4. Comparison:
   - How does the preprocessing of the C4 dataset compare to the preprocessing of the CommonCrawl dataset?

**For overall understanding**

1. Summary related queries:
   - Could you summarize the main data sources used for pre-training?

**To predict**

1. Potential Application and Consequence:
   - How could the use of this diverse set of training data potentially influence the performance and capabilities of the resulting language models?

2. Hypotheticals ("What if" scenarios):
   - How might the model performance change if the authors used a different tokenization technique instead of byte-pair encoding (BPE)?

3. Decision Justification:
   - Why did the authors choose to use the SwiGLU activation function instead of ReLU?