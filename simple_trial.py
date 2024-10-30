import os, sys
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/InternEmbedding')

import torch
import json
import numpy as np
from InternEmbedding.embedding.eval.metrics import cosine_similarity as matrix_cosine_similarity
from InternEmbedding.embedding.train.training_embedder import initial_model
from InternEmbedding.embedding.eval.mteb_eval_wrapper import EvaluatedEmbedder

class BGEEmbedder:
    def __init__(self, args):
        """Initialize the BGE embedder using InternEmbedding framework.
        
        Args:
            args: Configuration object that should contain:
                - device: str, device to run model on ('cuda:0', 'cuda:1', etc.)
                - embedder_ckpt_path: str, path to model checkpoint
                - max_length: int, maximum sequence length
        """
        # Initialize the base model and tokenizer
        embedder, tokenizer = initial_model(args)
        embedder = embedder.to(args.device)
        
        # Load checkpoint if provided
        if args.embedder_ckpt_path and os.path.exists(args.embedder_ckpt_path):
            embedder.load_state_dict(torch.load(args.embedder_ckpt_path))

        embedder.eval()
        
        # Create evaluated embedder wrapper
        self.embedder = EvaluatedEmbedder(
            embedder, 
            tokenizer, 
            args.max_length, 
            args.device
        )
        
        print(f'>>> Embedder has loaded in {args.device} successfully!')

    def encode(self, texts, batch_size=32):
        """Encode texts into embeddings.
        
        Args:
            texts (str or List[str]): Input text or list of texts
            batch_size (int): Batch size for processing
            
        Returns:
            numpy.ndarray: Text embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Use the batch_encode method from EvaluatedEmbedder
        embeddings = self.embedder.batch_encode(texts, batch_size=batch_size)
        embeddings = embeddings.cpu().numpy()
        return embeddings

    def find_most_similar(self, query_embedding, candidate_embeddings):
        """Find the index of the most similar embedding from a set of candidates.
        
        Args:
            query_embedding (numpy.ndarray): Query embedding of shape (embed_dim,)
            candidate_embeddings (numpy.ndarray): Matrix of candidate embeddings of shape (n_candidates, embed_dim)
            
        Returns:
            int: Index of the most similar embedding
        """
        # Ensure query_embedding is 2D for matrix operations
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Calculate dot product
        dot_product = np.dot(candidate_embeddings, query_embedding.T).squeeze()
        
        # Calculate magnitudes
        query_norm = np.sqrt(np.sum(query_embedding**2))
        candidate_norms = np.sqrt(np.sum(candidate_embeddings**2, axis=1))
        
        # Calculate cosine similarity
        similarities = dot_product / (query_norm * candidate_norms)
        
        # Get the index of the highest similarity scores
        most_similar_idx = np.argsort(similarities)[::-1]
        
        return most_similar_idx


from dataclasses import dataclass

# Create a config class similar to your yaml config
@dataclass
class EmbedderConfig:
    embedder_name: str = "bge_base15_embedder"
    backbone_type = "BGE"
    # init_backbone: BAAI/bge-base-en-v1.5
    init_backbone = "/cpfs01/shared/public/wangyikun/ckpt/embedding_model/bge-base-en-v1.5"
    flashatt = False
    pool_type = "cls"
    peft_lora = False
    which_layer = -1
    max_length = 512
    task_prompt = False
    checkpoint_batch_size = -1
    mytryoshka_size = 768
    embedding_norm = False
    embedder_ckpt_path = None
    reserved_layers = None
    device = "cuda"
    encode_batch_size = 2048
    data_path = "/cpfs01/shared/public/wangyikun/code/lctx_workspace/LongMIT/assets/example_datasets/custom_text_corpus.jsonl"
    output_npy = "./custom_text_corpus.npy"

# Initialize the embedder
config = EmbedderConfig()
embedder = BGEEmbedder(config)

# load data and parse jsonl
with open(config.data_path, 'r') as f:
    data = f.readlines()
data = [json.loads(line) for line in data]

# load encoded embeddings, embeddings should be a numpy array of shape (n_samples, embed_dim), type = np.float32
if os.path.exists(config.output_npy):
    embeddings = np.load(config.output_npy, allow_pickle=True)
else:
    # encode data
    data = [item['content'] for item in data]
    embeddings = embedder.encode(data)
    np.save(config.output_npy, embeddings)

# Use the embedder
# texts = ["a comprehensive article about Chopsticks, covering several key aspects"]
texts = ['A article about deep frying is a cooking method where food is fully submerged in hot oil, cooking all sides simultaneously. This ancient technique became widely popular in the 19th century.']
query_embedding = embedder.encode(texts)
query_embedding = query_embedding.reshape(1, -1)

# recall the most similar text
most_similar_idx = embedder.find_most_similar(query_embedding, embeddings)
top_indices = most_similar_idx[:3].tolist()
# print([data[idx] for idx in top_indices])

# print the most similar texts
for idx in top_indices:
    print('--'*10, f'idx: {idx}', '--'*10)
    print()
    print()
    print(data[idx]['content'])
    print()
    print()

