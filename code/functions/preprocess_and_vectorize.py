import spacy
import torch

nlp = spacy.load("en_core_web_lg") # if this fails then run "python -m spacy download en_core_web_lg" to download that model

def preprocess_and_vectorize(text, target):

    class_to_label = {
                      'Positive': torch.tensor(0),
                      'Negative': torch.tensor(1),
                      'Neutral': torch.tensor(2),
                      'Irrelevant': torch.tensor(3)
                      }

    doc = nlp(text)
    filtered_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]

    if len(filtered_tokens) == 0:
        return torch.from_numpy(nlp('UNK').vector), class_to_label[target]
    else:
        return torch.from_numpy(nlp(' '.join(filtered_tokens)).vector), class_to_label[target]