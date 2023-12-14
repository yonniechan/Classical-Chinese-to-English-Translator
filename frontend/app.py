from flask import Flask, request, render_template
import os
import argparse
import warnings
import torch
from transformers import AutoTokenizer, EncoderDecoderModel
from googletrans import Translator

app = Flask(__name__)
translator = Translator()

def inference(text, tokenizer, model):
    kwargs = dict(truncation=True, max_length=128, padding="max_length", return_tensors='pt')
   
    inputs = tokenizer([text,],**kwargs)
    
    with torch.no_grad():
        output = tokenizer.batch_decode(
            model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                num_beams=4,
                max_length=256,
                bos_token_id=101,
                eos_token_id=tokenizer.sep_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ),
            skip_special_tokens=True)
        return output[0].replace(' ', '')

@app.route('/', methods=['GET', 'POST'])
def translate():
    if request.method == 'POST':
        text = request.form['text']
        current_path = os.getcwd()
        model_path = os.path.join(current_path, "model")

        if not os.path.isdir(model_path):
            raise FileNotFoundError("The 'model' does not exist.")
        
        if text is None:
            raise ValueError("The text to translate should not be empty.")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = EncoderDecoderModel.from_pretrained(model_path)
        chinese_translation = inference(text, tokenizer, model)
        eng_translation = translator.translate(chinese_translation, dest='en').text
        return render_template('index.html', text=text, chinese_translation=chinese_translation, eng_translation=eng_translation)
    return render_template('index.html')

if __name__ == "__main__":
     app.run(debug=True, host='0.0.0.0', port=7060)
