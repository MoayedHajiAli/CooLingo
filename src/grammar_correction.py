from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class GrammarCorrector():

    def __init__(self):
        # The __init__ initializes the model,
        # NOTE: Run this in a separate cell becuase it takes 7 seconds
        # to load the model
        self.tokenizer = AutoTokenizer.from_pretrained("addy88/t5-grammar-correction")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("addy88/t5-grammar-correction")

    def correct_grammar(self, input, token='grammar'):
        """
        Takes an input sentence and returns the a corrected version of it
        if there is any grammatical error in the input sentence.
        
        Parameters:
            input (str): An input sentence to be checked for grammatical errors.

        Returns: A grammatically corrected version of the input sentence.
        """
        input_ids = self.tokenizer(f'{token}: {input}', return_tensors='pt').input_ids
        outputs = self.model.generate(input_ids)
        
        corrected_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return corrected_output
