import os
import argparse
import warnings
import torch
from transformers import AutoTokenizer, EncoderDecoderModel

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

def main(params):
    current_path = os.getcwd()
    model_path = os.path.join(current_path, "model")

    if not os.path.isdir(model_path):
        raise FileNotFoundError("The 'model' does not exist.")
    
    if params.text is None:
        raise ValueError("The text to translate should not be empty.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    print(inference(params.text, tokenizer, model))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Language Model")

    parser.add_argument("--text", type=str, default=None)

    params, unknown = parser.parse_known_args()
    
    main(params)

