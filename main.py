import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


model_name = "dkleczek/Polish-Hate-Speech-Detection-Herbert-Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
hate_speech_classifier = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

hate_speech_classifier.load_state_dict(torch.load('hate_classifier.pt'))

def tokenize_function(data):
    return tokenizer(data, padding=True, truncation=True)


def is_hate_speech(text):
    tokenized_text = tokenize_function(text)
    inputs_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    batch = {'attention_mask': torch.tensor([attention_mask]), "input_ids": torch.tensor([inputs_ids])}
    with torch.no_grad():
        result = hate_speech_classifier(**batch).logits
        prediction = torch.argmax(result, dim=-1)
    if prediction.item() == 0:
        return False
    return True


def main():
    st.title("Hate speech detector")
    with st.form(key='emotion_clf_form'):
        text_input = st.text_area(label='Enter a comment 👇')
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        result = is_hate_speech(text_input)
        if result is True:
            st.markdown("<h1 style='text-align: center'>This is a hate speech 🤬</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center'>This is not a hate speech 😊</h1>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
