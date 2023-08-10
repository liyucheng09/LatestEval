# This file defines DataProcessors, for each dataset.

from datasets import load_dataset
import re
from utils import Doc
from nltk.tokenize import sent_tokenize
import os
import openai
import time
from collections import Counter
import markdown
from bs4 import BeautifulSoup


class DataProcessor:
    """
    Base class for data processors.

    Data processors are used to:
    - load data from files or huggingface datasets
    - seperate each document into fragments: answer fragments and context fragments
        we use simple rule-based method to seperate fragments
    - use answer identifier to identify key points from the answer fragments
        we use ChatGPT like LLMs to identify key points. The key points are used as reference answers. So we copy the original text from the answer fragments directly.
    - apply query generation funcs on key points to generate queries
        we use ChatGPT like LLMs to generate queries
    - format the (query, answer, context) pairs
    """

    def __init__(self, source):
        self.source = source

        self.answer_templated = {
            'summary': [
                'in summary',
                'to summarize',
                'main contribution',
                'in conclusion',
                'major contribution',
                'in short',
                'td;dr',
            ],
            'purpose': [
                'because',
                'in order to',
                'allow us to',
                'lead to',
                'so that',
                'enable',
                'allow',
            ],
            'example': [
                'for example',
                'e.g.,',
                'for instance',
            ],
            'future': [
                'future',
                'upcoming feature',
                'forecast',
            ],
        }

        self.query_templates = {
            "term": [
                "What is {term}?",
                "Can you explain what {term} means?", 
                "What does the author mean by {term}?",
                "How does {term1} differ from {term2}?"
            ],
            
            "summary": [
                "Can you summarize this {section}?",
                "What are the key points made in this {section}?",
                "What is the main finding in this {section}?" 
            ],
            
            "purpose": [
                "What is the purpose of the {object}?",
                "Why did the author propose the {object}?",
                "What problem does the {object} aim to solve?"
            ],

            "example": [
                "Can you provide examples of {object}?",
                "What are some examples of {concept} in the passage?",
                "Could you illustrate {concept} with examples?"
            ],

            "future": [
                "What future work does the author suggest?",
                "How might {concept} evolve in the future according to the passage?", 
                "What predictions does the author make about {concept}?"
            ]
        }


    def prepare_docs(self):
        raise NotImplementedError
    
    def answers(self):
        """
        Identify key points from fragments via copying the original text.

        Args:
            fragment: the answer fragment
        """
        for doc in self.docs:
            if doc.answers_sentences is None:
                doc.answers_sentences = []
            for sent in doc.original_sentences:
                for categoty, templates in self.answer_templated.items():
                    for template in templates:
                        if template in sent.lower():
                            doc.answers_sentences.append((sent, categoty))
                            break
        
        all_types = []
        for doc in self.docs:
            for sent, category in doc.answers_sentences:
                all_types.append(category)
        print(Counter(all_types))
        
        # use ChatGPT/GPT-4 to extract the exact answer from the answer sentence
        file =  open('prompt.txt', 'w')
        for doc in self.docs:
            for sent, category in doc.answers_sentences:
                prompt = self.prompt_for_gpt('answer', sent, category)
                file.write(prompt+'\n')
                # answer = self.gpt(sent)
                # doc.answers.append(answer)
        
        file.close()
    
    def queries(self, key_point, fragment):
        """
        Generate queries from key points.

        Args:
            key_point: the key point
            fragment: the answer fragment
        """
        raise NotImplementedError
    
    def gpt(self, prompt, num_retry = 5):
        # generate answer by gpt-3.5-turbo
        openai_key = os.environ.get("OPENAI_API_KEY")
        for _ in range(num_retry):
            try:
                r = openai.ChatCompletion.create(
                    model = 'gpt-3.5-turbo',
                    messages = [
                        {"role": "user", "content": prompt},
                    ],
                    temperature = 0,
                )
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        
        return r.choices[0]['message']['content']
    
    def prompt_for_gpt(self, answer_or_query, answers_sentence, category):
        # generate prompt for gpt
        if answer_or_query == 'answer':
            leading = {
                'summary': 'The following sentence has a summative statement. Please extract and copy the summative content.',
            }
            prompt = f"{leading[category]}\n----\n{answers_sentence}"
        elif answer_or_query == 'query':
            leading = {
                'summary': f'The following sentence has a summary. Please generate a question targets the summative content. Here are some examples: {self.query_templates[category]}',
            }
            prompt = f"{leading[category]}\n----\n{answers_sentence}"
        
        return prompt
    
    def qa_pairs(self, query, answer, context):
        pairs = []
        for doc in self.docs:
            passage = doc.original_passage
            for answer, query in zip(doc.answers, doc.queries):
                pairs.append(
                    {
                        'query': query,
                        'answer': answer,
                        'passage': passage,
                    }
                )
        return pairs

class ArxivProcessor(DataProcessor):
    """
    Data processor for arxiv dataset.
    """

    def __init__(self, dataset_name):
        """
        Args:
            args: the arguments
        """
        super().__init__(source='arxiv')
        self.dataset_name = dataset_name
        self.prepare_docs()

    def prepare_docs(self):
        self.ds = load_dataset(self.dataset_name, split='train[:500]')

        self.docs = []
        docs = []
        for instance in self.ds:
            entry_id = instance['entry_id']
            text = instance['text']
            source = self.source

            doc = Doc(text=text, source=source, entry_id=entry_id)
            docs.append(doc)
        
        print(f"Loaded {len(self.docs)} documents from {self.source} on {self.dataset_name} time slice.")
        
        def parse_arxiv(doc):
            text = doc.text
            # remove anything before introduction
            # text = re.sub(r"^.*?(§)", r"\1", text, flags=re.DOTALL)

            # split article into sections
            sections = re.split(r"(?<!§\.)§\s", text)
            sections = [self.beautify_context(section) for section in sections if section.strip()]
            if len(sections) < 2:
                return None, sections
            
            passages = {}
            for i, section in enumerate(sections):
                if i ==0: 
                    passages['summary'] = f'Abstract\n{section}'
                if i ==1: 
                    passages['intro'] = section
                if section.lower().startswith('conclusion'):
                    passages['conclusion'] = section
                    continue
                
            return passages, sections

        for index, doc in enumerate(docs):
            fragments, sections = parse_arxiv(doc)
            if fragments is None: 
                continue
            doc.meta_data = fragments
            doc.original_passage = fragments['intro']
            doc.original_sentences = sent_tokenize(doc.original_passage)
            doc.sections = sections
            self.docs.append(doc)

    def beautify_context(self, context: str) -> str:
        context = context.replace("<cit.>", '').replace('<ref>', '')
        context = re.sub(r"\s+", " ", context)
        context = re.sub(r"\n+", " ", context)

        return context.strip()

class BBCNewsProcessor(DataProcessor):

    def __init__(self, dataset_name):
        """
        Args:
            args: the arguments
        """
        super().__init__(source='bbc')
        self.dataset_name = dataset_name
        self.prepare_docs()

    def prepare_docs(self):
        self.ds = load_dataset(self.dataset_name, split='train[:500]')

        self.docs = []
        docs = []
        for instance in self.ds:
            entry_id = instance['link']
            text = instance['content']
            source = self.source

            title = instance['title']
            description = instance['description']

            doc = Doc(text=text, source=source, entry_id=entry_id, meta_data={'title': title, 'description': description})
            docs.append(doc)
        
        print(f"Loaded {len(self.docs)} documents from {self.source} on {self.dataset_name} time slice.")

        for index, doc in enumerate(docs):
            if 'live' in doc.entry_id: 
                continue
            doc.original_passage = doc.text
            doc.original_sentences = sent_tokenize(doc.original_passage)
            self.docs.append(doc)

class GithubProcessor(DataProcessor):

    def __init__(self, dataset_name):
        """
        Args:
            args: the arguments
        """
        super().__init__(source='github')
        self.dataset_name = dataset_name
        self.prepare_docs()

    def prepare_docs(self):
        self.ds = load_dataset(self.dataset_name, split='train[:500]')

        self.docs = []
        for instance in self.ds:
            entry_id = instance['full_name']
            original_sentences = self.markdown_to_plain_text(instance['readme'])
            text = '\n'.join(original_sentences)
            source = self.source

            description = instance['description']

            doc = Doc(text=text, source=source, entry_id=entry_id, meta_data={'description': description}, original_sentences=original_sentences, original_passage=text)
            self.docs.append(doc)
        
        print(f"Loaded {len(self.docs)} documents from {self.source} on {self.dataset_name} time slice.")

    def markdown_to_plain_text(self, md_text):
        # Add newline before list items that are denoted by " - "
        md_text = md_text.replace(" - ", "\n- ")
        
        # Convert markdown to HTML
        html_text = markdown.markdown(md_text)
        text = ''.join(BeautifulSoup(html_text, features="html.parser").stripped_strings)

        # Replace markdown headers with a newline
        text = re.sub(r'#+\s', '\n', text)
        
        # Remove emoji-like patterns
        text = re.sub(r':\w+:', '', text)
        
        # Add newlines around numbers likely to be part of a list
        text = re.sub(r'(\d+\.)', r'\n\1', text)

        return [line.strip() for line in text.strip().split('\n') if line.strip()]

if __name__ == '__main__':
    # processor = ArxivProcessor('RealTimeData/arxiv_july_week1_2023')
    # processor.answers()

    # processor = BBCNewsProcessor('RealTimeData/bbc_news_week1_july_2023')
    # processor.answers()

    processor = GithubProcessor('RealTimeData/github_july_week1_2023')
    processor.answers()
