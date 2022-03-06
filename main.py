import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@st.cache
def prepare_model(model_name):
    hate_speech_classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    return hate_speech_classifier


@st.cache
def load_model(model):
    model.load_state_dict(torch.load('hate_classifier.pt'))


def tokenize_function(data, tokenizer):
    return tokenizer(data, padding=True, truncation=True)


def is_hate_speech(text, model, tokenizer):
    tokenized_text = tokenize_function(text, tokenizer)
    inputs_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    batch = {'attention_mask': torch.tensor([attention_mask]), "input_ids": torch.tensor([inputs_ids])}
    with torch.no_grad():
        result = model(**batch).logits
        prediction = torch.argmax(result, dim=-1)
    if prediction.item() == 0:
        return False
    return True


def main():
    model_name = "dkleczek/Polish-Hate-Speech-Detection-Herbert-Large"
    hate_speech_classifier = prepare_model(model_name)
    load_model(hate_speech_classifier)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    st.title("Hate speech detector")
    with st.form(key='emotion_clf_form'):
        text_input = st.text_area(label='Enter a comment 👇')
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        result = is_hate_speech(text_input, hate_speech_classifier, tokenizer)
        if result is True:
            st.markdown("<h1 style='text-align: center'>This is a hate speech 🤬</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center'>This is not a hate speech 😊</h1>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
