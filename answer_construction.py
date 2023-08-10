from nltk.tokenize import sent_tokenize

class AnswerProcessor:

    def __init__(self, type_of_query):
        self.type_of_query = type_of_query
    
    def get_answers(self):
        raise NotImplementedError("get_answers method not implemented")
    
    def retrieve_relevant_sentences(self):
        raise NotImplementedError("retrieve_relevant_sentences method not implemented")

class TerminologyAnswerProcessor(AnswerProcessor):

    def __init__(self, type_of_query):
        super().__init__(type_of_query)
    
    def get_answers(self):
        return self.terminology
    
    def retrieve_relevant_sentences(self):
        return None


class SummaryAnswerProcessor(AnswerProcessor):

    def __init__(self, type_of_query):
        super().__init__(type_of_query)

    def get_answers(self, docs):
        

    def retrieve_relevant_sentences(self):
        return None