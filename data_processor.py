# This file defines DataProcessors, for each dataset.

from datasets import load_dataset
import re

class DataProcessor:
    """
    Base class for data processors.

    Data processors are used to:
    - load data from files
    - seperate each document into fragments: answer fragments and context fragments
        we use simple rule-based method to seperate fragments
    - use answer identifier to identify key points from the answer fragments
        we use ChatGPT like LLMs to identify key points. The key points are used as reference answers. So we copy the original text from the answer fragments directly.
    - apply query generation funcs on key points to generate queries
        we use ChatGPT like LLMs to generate queries
    - format the (query, answer, context) pairs
    """

    def fragments(self, doc):
        """
        Seperate a document into fragments.

        Args:
            doc: a document

        Returns:
            a dict of fragments, now it will be answer fragments and context fragments
        """
        raise NotImplementedError
    
    def key_points(self, fragment):
        """
        Identify key points from fragments via copying the original text.

        Args:
            fragment: the answer fragment
        """
        raise NotImplementedError
    
    def queries(self, key_point, fragment):
        """
        Generate queries from key points.

        Args:
            key_point: the key point
            fragment: the answer fragment
        """
        raise NotImplementedError
    
    def format(self, query, answer, context):
        """
        Format the (query, answer, context) pairs.

        Args:
            query: the query
            answer: the answer
            context: the context
        """
        raise NotImplementedError

class ArxivProcessor(DataProcessor):
    """
    Data processor for arxiv dataset.
    """

    def __init__(self, dataset_name):
        """
        Args:
            args: the arguments
        """
        self.ds = load_dataset(dataset_name, split='train')
    
    def beautify_context(self, context: str) -> str:
        context = context.replace("<cit.>", '').replace('<ref>', '')
        context = re.sub(r"\s+", " ", context)
        context = re.sub(r"\n+", " ", context)

        return context.strip()
    
    def fragments(self):
        """
        Seperate a document into fragments.

        Args:
            doc: a document

        Returns:
            a dict of fragments, now it will be answer fragments and context fragments
        """
        # seperate the document into fragments
        # we use the simple rule-based method to seperate fragments
        # we take the intro, abstract, and conclusion as the answer fragment
        # the rest sentences as the context fragment

        def parse_arxiv(doc):
            text = doc['text']

            # remove anything before introduction
            # text = re.sub(r"^.*?(§)", r"\1", text, flags=re.DOTALL)

            # split article into sections
            sections = re.split(r"(?<!§\.)§\s", text)
            sections = [self.beautify_context(section) for section in sections if section.strip()]
            
            ans_fragments = []
            context_fragments = []
            for i, section in enumerate(sections):
                # the first section is usually the abstract, second is the intro
                if i in [0, 1]:
                    if i == 0: section = f'Abstract\n{section}'
                    ans_fragments.append(section)
                    continue
                # the last section is usually the conclusion, but sometimes acknowledgements, so we need to check
                if section.lower().startswith('conclusion'):
                    ans_fragments.append(section)
                    continue
                
                context_fragments.append(section)

            ans_fragment = '\n\n'.join(ans_fragments)
            context_fragment = '\n\n'.join(context_fragments)

            print(sections)
            return {
                'ans': ans_fragment,
                'context': context_fragment
            }

        ds = self.ds.map(parse_arxiv)
    
    def key_points(self, fragment):
        """
        Identify key points from fragments via copying the original text.

        Args:
            fragment: the answer fragment
        """
        # key points are the original text of the answer fragment
        key_points = fragment
        return key_points
    
    def queries(self, key_point, fragment):
        """
        Generate queries from key points.

        Args:
            key_point: the key point
            fragment: the answer fragment
        """
        # generate queries from key points
        # we use the simple rule-based method to generate queries
        # we use the first sentence of the answer fragment as the query
        queries = fragment[0]
        return queries
    
    def format(self, query, answer, context):
        """
        Format the (query, answer, context) pairs.

        Args:
            query: the query
            answer: the answer
            context: the context
        """
        # format the (query, answer, context) pairs
        # we use the simple rule-based method to format the pairs
        # we use the first sentence of the answer fragment as the query
        # we use the first sentence of the answer fragment as the answer
        # we use the rest sentences of the answer fragment as the context
        formatted = {
            'query': query[0],
            'answer': answer[0],
            'context': context[1:]
        }
        return formatted

if __name__ == '__main__':
    processor = ArxivProcessor('liyucheng/arxiv-march-2023')
    processor.fragments()
