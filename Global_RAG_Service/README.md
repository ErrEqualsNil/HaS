# Full-Corpus Retrieval Service: Cloud Simulation

Due to **limited computational resources**, all query retrieval results were **pre-computed** offline.

To simulate the **online retrieval service** performance, we executed **10,000** retrieval operations to compute the mean ($\mu$) and standard deviation ($\sigma$) of the retrieval latency.

The online service's latency is simulated by **sampling from a normal distribution** $\mathcal{N}(\mu, \sigma)$.


### To start this service, please fill the following path in Retriever.py

precomputed_path: pre-computed retrieval results

embedding_dir: pre-computed embedding of all documents in the corpus

ids_idr: indices of all documents in the corpus

model_name: semantic encoder used for retrieval
