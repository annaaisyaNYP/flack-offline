from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModelForTokenClassification
import tensorflow as tf
import numpy as np


model_checkpoint = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
intent_model = TFAutoModelForSequenceClassification.from_pretrained(
    './intent_model_a1.0/')
token_model = TFAutoModelForTokenClassification.from_pretrained('./token_model/')
sentence_labels = ["Others", "Define", "Locate", "Can-I", "Points"]
token_labels = ['O', 'B-PER', 'I-PER', 'B-ORG',
                'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']


def infer_intent(text, model, tokenizer):
    input = tokenizer(text, truncation=True, padding=True, return_tensors="tf")
    output = model(input)
    pred_label = np.argmax(tf.nn.softmax(output.logits, axis=-1))
    return sentence_labels[pred_label]


def infer_tokens(text, model, tokenizer):
    text = text.split()

    encodings = tokenizer(
        [text],
        padding=True,
        truncation=True,
        is_split_into_words=True,
        return_tensors='tf')

    logits = model(encodings)[0]
    preds = np.argmax(logits, axis=-1)[0]

    previous_word_idx = None
    word_ids = encodings[0].word_ids
    labels = []
    for i, word_idx in enumerate(word_ids):
        if word_idx != previous_word_idx and word_idx != None:
            labels.append(token_labels[preds[i]])
        previous_word_idx = word_idx

    return text, labels


def infer_combined_tokens(text, token_model, tokenizer):
    result = {
        "PER": [],
        "LOC": [],
        "ORG": [],
        "MISC": []
    }

    result_texts, result_tokens = infer_tokens(text, token_model, tokenizer)

    current_token_label = ""
    current_result_index = -1

    for i in range(len(result_tokens)):
        if result_tokens[i].startswith("B-"):
            current_token_label = result_tokens[i].replace("B-", "")
            result[current_token_label].append(result_texts[i])
            current_result_index = len(result[current_token_label]) - 1
        elif result_tokens[i].startswith("I-"):
            result[current_token_label][current_result_index] += " " + \
                result_texts[i]

    return result

def result(input):

    if isinstance(input, str):
        intent = infer_intent(input,intent_model,tokenizer)
    else:
        intent = infer_intent(input["text"],intent_model,tokenizer)

    if (intent == "Define"):
        response = "Chatbot: To know more about E-waste, click here."
    elif intent == "Locate":
        response = "Chatbot: If you are looking for the nearest E-waste bin, click here to find out." 
    elif intent == "Can-I":
        response = "Chatbot: To learn more about what e-waste you can recycle, click here."
    elif intent == "Points":
        response = "Chatbot: If you want to know more about the rewards program, click here to find out."       
    else:
            response = "I don't quite know how to respond to " + intent + " yet."

    return response
