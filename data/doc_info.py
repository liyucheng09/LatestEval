import docx
import re
from wikipedia import gpt3_self_info
import sys

def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def beautify_text(text, num_words = 1000):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    
    # use first 1000 words
    text = ' '.join(text.split(' ')[:num_words])
    return text

def verbalise_docs():
    docs = ['data/q17-1.docx', 'data/q18-1.docx', 'data/q19-1.docx', 'data/q20-1.docx', 'data/q22-1.docx', 'data/q23-1.docx']
    doc_text = [ getText(doc) for doc in docs ]

    doc_text = [ beautify_text(doc) for doc in doc_text ]

    return {
        doc: [doc_string] for doc, doc_string in zip(docs, doc_text)
    }

if __name__ == '__main__':
    docs = ['data/q17-1.docx', 'data/q18-1.docx', 'data/q19-1.docx', 'data/q20-1.docx', 'data/q22-1.docx', 'data/q23-1.docx']
    doc_text = [ getText(doc) for doc in docs ]

    doc_text = [ beautify_text(doc) for doc in doc_text ]

    for doc, doc_string in zip(docs, doc_text):
        print('----------------------')
        print(doc)

        _, info = gpt3_self_info(doc_string)
        print(sum(info)/len(info))