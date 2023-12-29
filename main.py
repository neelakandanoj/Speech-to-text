#Import Libraries
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from evaluate import load
import json


# to check using GPU or Not
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

#Main Method
def get_transcription(filename: str):
    global sample
    
    #Model using the pretrained openai whisper large model to convert the audio file into the text format
    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, use_safetensors=True)
    model.to(device)
    #Processor using the necessary parameters
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,)
    sample = filename
    transcription = pipe(sample)
    return transcription

# Print or work with the loaded dat
if __name__ == "__main__":
   file=input("enter the file name: ")
   result=get_transcription(file)
   print(result['text'])
