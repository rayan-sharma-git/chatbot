# step-1 imports
import os
import json
import faiss
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# step-2 data loading
def load_all_json(folder_path):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith('.json'):
            with open(os.path.join(folder_path, file), 'r', encoding = 'utf-8') as f:
                data = json.load(f)

                for item in data:
                    text = f'''
Type: {item.get('type')}
Exam: {item.get('exam')}
Subject: {item.get('subject')}
Year: {item.get('year')}
Section: {item.get('section')}
Marks: {item.get('marks')}

Question: {item.get('question')}
Answer: {item.get('answer')}
'''
                    documents.append(text.strip())
 
    return documents

documents = load_all_json('data')
print('Documents Loaded: ', len(documents))

# step-3 Text splitting
# since we have the data in json, we dont really need to split the data into chunks
chunks = documents

# step-4 create and store embeddings
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

index_path = 'faiss_index'

if os.path.exists(index_path):
    print('Loading existing FAISS index...')
    index = faiss.read_index(os.path.join(index_path, 'index.faiss'))
    with open(os.path.join(index_path, 'docs.json'), 'r', encoding='utf-8') as f:
        chunks = json.load(f)
else:
    print('Creating new FAISS index...')
    
    embeddings = embed_model.encode(chunks)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, 'index.faiss'))
    
    with open(os.path.join(index_path, 'docs.json'), 'w', encoding='utf-8') as f:
        json.dump(chunks, f)

# step-5 creating Retriever
def retrieve(query, top_k = 3, similarity_multiplier=1.5, max_tokens=2048):
    
    query_embedding = embed_model.encode([query])
    
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    distances = distances[0]
    indices = indices[0]
    
    best_distance = distances[0]
    
    selected_docs = []
    total_tokens = 0
    
    for dist, idx in zip(distances, indices):
        
        if dist > best_distance * similarity_multiplier:
            break
        
        doc = chunks[idx]
        
        doc_tokens = len(doc.split())
        
        if total_tokens + doc_tokens > max_tokens:
            break
        
        selected_docs.append(doc)
        total_tokens += doc_tokens
    
    return selected_docs

# step-6 llm
from peft import PeftModel

base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
lora_model = "./saved_model/tiny_llama/final_adapter"

tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    model,
    lora_model
)

print("TinyLlama + LoRA loaded")

# step-7 prompt
template = '''Context:
{context}

Question:
{question}

If the question says 1 mark, then answer in 40 to 60 words.
If the question says 2 marks, then answer in 60 to 80 words.
If the question says 5 marks, then answer it 350 to 400 words.
Try to answer in bullet points.

ANSWER:
'''

# step-8 format retrieved docs
def format_docs(docs):
    return '\n\n'.join(docs)

# step-9 rag pipeline
def rag_chain(query):
    
    docs = retrieve(query)
    context = format_docs(docs)
    
    prompt = template.format(context=context, question=query)
    
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    
    outputs = model.generate(
    **inputs,
    max_new_tokens = 700,
    temperature = 0.3,
    do_sample = True,
    repetition_penalty = 1.2,
    pad_token_id = tokenizer.eos_token_id,
    eos_token_id = tokenizer.eos_token_id
)
    
    input_length = inputs['input_ids'].shape[1]

    generated_tokens = outputs[0][input_length:]

    final_answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens = True
        ).strip()

    
    return final_answer


# step-10 ask question
if __name__ == '__main__':

    while True:

        query = input('\nAsk a question (or type "exit"): ')

        if query.lower().strip() == 'exit':
            break

        response = rag_chain(query)

        print('\nAnswer:\n')
        print(response)

# end of code