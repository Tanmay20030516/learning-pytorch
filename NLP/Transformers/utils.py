import torch
import spacy
from torcheval.metrics.functional import bleu_score
import sys
import numpy as np
import matplotlib.pyplot as plt


def translate_sentence(model, sentence, english, german, device, max_length=50):
    # print(sentence)

    # sys.exit()

    # Load english tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_eng(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(english.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [english.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [german.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if best_guess == german.vocab.stoi["<eos>"]:
            break

    translated_sentence = [german.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, english, german, device):
    """
    Evaluating the quality of machine-translated text using BLEU score.
    :param data: Test data
    :param model: The trained model
    :param english: English language field in torchtext
    :param german: German language field in torchtext
    :param device: Device to run the model on (e.g., "cuda" or "cpu")
    :return: BLEU score
    """
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        # Convert the list of tokens to a space-separated string
        prediction_str = " ".join(prediction)

        targets.append(trg)  # Remove the inner list, assuming trg is a string
        outputs.append(prediction_str)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
