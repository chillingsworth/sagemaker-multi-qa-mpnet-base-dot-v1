from sagemaker.huggingface import HuggingFaceModel
import sagemaker 

role = 'arn:aws:iam::484906661071:role/sagemaker'

# Hub Model configuration. https://huggingface.co/models
hub = {
  'HF_MODEL_ID':'multi-qa-mpnet-base-dot-v1', # model_id from hf.co/models
  'HF_TASK':'sentence-similarity' # NLP task you want to use for predictions
}

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   env=hub,
   role=role, # iam role with permissions to create an Endpoint
   transformers_version="4.6", # transformers version used
   pytorch_version="1.7", # pytorch version used
   py_version="py36", # python version of the DLC
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
   initial_instance_count=1,
   instance_type="ml.m5.xlarge"
)

print(predictor)

# example request, you always need to define "inputs"
data = {
"inputs": {
    "source_sentence": "What is used for inference?",
    "sentences": ["My Name is Philipp and I live in Nuremberg.", "This model is used with sagemaker for inference."]
    }
}

# request
print(predictor.predict(data))

# # pipe = pipeline("sentence-similarity", model="multi-qa-mpnet-base-dot-v1", device=0)
# pipe = pipeline("sentiment-analysis")
query = "How many people live in London?"
docs = ["Around 9 Million people live in London", "London is known for its financial district"]

# #Encode query and documents
# query_emb = model.encode(query)
# doc_emb = model.encode(docs)

# #Compute dot score between query and all document embeddings
# scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

# #Combine docs & scores
# doc_score_pairs = list(zip(docs, scores))

# #Sort by decreasing score
# doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

# #Output passages & scores
# for doc, score in doc_score_pairs:
#     print(score, doc)