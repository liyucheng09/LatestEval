import arxiv
import datetime
from queue import Queue
from threading import Thread, Lock
import os
import logging
import time
import tarfile
from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexMacroNode
from pylatexenc import latex2text
from pylatexenc.macrospec import LatexContextDb
import shutil
import re
import json
from glob import glob
from huggingface_hub import create_branch, create_tag, RepoCard
import datasets
import sys

def filter_element(context, exclude_elements = []):
    
    new_context = LatexContextDb()

    new_context.unknown_macro_spec = context.unknown_macro_spec
    new_context.unknown_environment_spec = context.unknown_environment_spec
    new_context.unknown_specials_spec = context.unknown_specials_spec

    filter_element_func = lambda dict_to_filter: {k:v for k,v in dict_to_filter.items() if k not in exclude_elements}.values()
    for cat in context.category_list:

        # include this category
        new_context.add_context_category(
            cat,
            macros=filter_element_func(context.d[cat]['macros']),
            environments=filter_element_func(context.d[cat]['environments']),
            specials=filter_element_func(context.d[cat]['specials']),
        )

    return new_context

class TextExtractor:

    def __init__(self):
        self.l2t_context_db = latex2text.get_default_latex_context_db()
        self.l2t_context_db.add_context_category(
            'Abstract',
            macros={},
            environments=[
                latex2text.EnvironmentTextSpec("abstract", simplify_repl=r'§ ABSTRACT %(body)s'),
                latex2text.EnvironmentTextSpec("Abstract", simplify_repl=r'§ ABSTRACT %(body)s')
            ],
            specials={}
        )
        self.l2t_context_db = filter_element(self.l2t_context_db, ['href'])

        self.l2t = latex2text.LatexNodes2Text(latex_context=self.l2t_context_db)
    
    def extract(self, latex_code):
        result = parse_tex_ignore_figures(latex_code)
        return self.l2t.nodelist_to_text(result)

def remove_figure_nodes(node_list):
    filtered_node_list = []
    for node in node_list:
        # Ignore the 'figure' environment
        if node.isNodeType(LatexEnvironmentNode):
            if node.environmentname in [ 'figure', 'figure*', 'algorithm', 'table', 'table*', 'algorithmic']:
                continue
        if hasattr(node, 'nodelist'):
            node.nodelist = remove_figure_nodes(node.nodelist)
        filtered_node_list.append(node)
    return filtered_node_list

def parse_tex_ignore_figures(tex_code):
    walker = LatexWalker(tex_code)
    parsed = walker.get_latex_nodes()[0]

    for node in parsed:
        if node.isNodeType(LatexEnvironmentNode):
            if node.environmentname == 'document':
                parsed = [node]
                break

    filtered_nodes = remove_figure_nodes(parsed)
    return filtered_nodes

def resolve_input_commands(latex_code, base_dir="."):
    input_pattern = re.compile(r"(?<!\\)\\input\{(.*?)\}")
    comment_pattern = re.compile(r"(?<!\\)%.*")

    def replace_input(match):
        filename = match.group(1)
        file_path = os.path.join(base_dir, filename)
        if not file_path.endswith(".tex"):
            file_path += ".tex"
        with open(file_path, "r", encoding='utf-8', errors='ignore') as input_file:
            content = input_file.read()
        return resolve_input_commands(content, base_dir=os.path.dirname(file_path))

    # Remove comments
    code_no_comments = comment_pattern.sub("", latex_code)

    # Resolve input commands
    resolved_code = input_pattern.sub(replace_input, code_no_comments)

    return resolved_code

def pruned_latex_to_text(latex_code, math_mode = 'remove'):
    result = parse_tex_ignore_figures(latex_code)
    return latex2text.LatexNodes2Text(math_mode = math_mode).nodelist_to_text(result)

class Worker(Thread):
    def __init__(self, queue, thread_id, text_save_dir):
        Thread.__init__(self)
        self.queue = queue
        self.thread_id = thread_id
        self.text_save_dir = text_save_dir

        self.text_extractor = TextExtractor()

        # Initialize logging for this thread
        self.logger = logging.getLogger(f"Thread-{thread_id}")
        handler = logging.FileHandler(f"thread_{thread_id}.log")
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def run(self):
        while True:
            # Get the work from the queue and expand the tuple
            result_index, result = self.queue.get()
            self.process_result(result)
            self.queue.task_done()
    
    def process_result(self, result):
        meta_data = {
            'entry_id': result.entry_id,
            'published': result.published.strftime("%Y%m%d%H%M%S"),
            'title': result.title,
            'authors': [author.name for author in result.authors],
            'primary_category': result.primary_category,
            'categories': result.categories
        }
        # self.logger.info(f'------ META {meta_data}')
        paper_id = result.entry_id.split('/')[-1]
        try:
            result.download_source('./', filename = f'{paper_id}.arxiv_source')
        except Exception as e:
            self.logger.error(f'ERROR: {e}')
            time.sleep(3)
            return
        
        try:
            with tarfile.open(f'{paper_id}.arxiv_source') as tar:
                tar.extractall(f'./{paper_id}')
            logging.info(f'------ Extracted {paper_id}.arxiv_source')
        except Exception as e:
            os.remove(f'{paper_id}.arxiv_source')
            self.logger.error(f'ERROR: {e}')
            time.sleep(3)
            return
        
        try:
            extracted_files = os.listdir(f'./{paper_id}')
            tex_files = [file for file in extracted_files if file.endswith('.tex')]
            if len(tex_files) > 1:
                if 'main.tex' in tex_files: tex_files = ['main.tex']
                else:
                    self.logger.info(f'------ Found multiple tex files: {tex_files}')
                    return
            elif len(tex_files) == 0:
                self.logger.info(f'------ Found no tex files')
                return
            tex_file = tex_files[0]
            with open(f'./{paper_id}/{tex_file}', 'r', encoding='utf-8', errors='ignore') as f:
                latex_code = f.read()
                if '\\input' in latex_code:
                    latex_code = resolve_input_commands(latex_code, base_dir=f'./{paper_id}')
                text = self.text_extractor.extract(latex_code)
            
            meta_data['text'] = text
            with open(f'{self.text_save_dir}/{paper_id}.json', 'w') as f:
                json.dump(meta_data, f, ensure_ascii=False)
            
            self.logger.info(f'------ Saved {paper_id}.json')
            
        except Exception as e:
            self.logger.error(f'ERROR: {e}')
            time.sleep(3)
            return

        finally:
            shutil.rmtree(f'./{paper_id}')
            os.remove(f'{paper_id}.arxiv_source')


if __name__ == '__main__':
    today = datetime.date.today()
    year = today.year
    month = today.month
    save_dir = './arxiv_data/'

    hf_token = os.environ['HF_TOKEN']
    time_stamp = f'{year}-{month:02d}'

    first_day = datetime.date(int(year), int(month), 1)
    last_day = datetime.date(int(year), int(month), 28)

    start_time_str = first_day.strftime("%Y%m%d%H%M%S")
    end_time_str = last_day.strftime("%Y%m%d%H%M%S")

    text_save_dir = os.path.join(save_dir, time_stamp)
    if not os.path.exists(text_save_dir):
        os.makedirs(text_save_dir)
    
    search = arxiv.Search(
        query=f'submittedDate:[{start_time_str} TO {end_time_str}]',
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=1000
    )

    q = Queue()
    num_threads = 4
    
    for i in range(num_threads):
        worker = Worker(q, i, text_save_dir,)
        worker.daemon = True
        worker.start()

    for index, result in enumerate(search.results()):
        q.put((index, result))

    q.join()

    print(f"Finished {time_stamp}")

    files = glob(f'{text_save_dir}/*.json')
    ds = datasets.load_dataset('json', data_files=files, split='train')

    ds.push_to_hub(
        "RealTimeData/arxiv_alltime",
        config_name=time_stamp,
        token=hf_token,
    )