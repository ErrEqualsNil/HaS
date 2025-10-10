# Speculative Retrieval Service: Edge Simulation

Due to **limited computational resources**, all query retrieval results were **pre-computed** offline.

To simulate the **online retrieval service** performance, we executed **10,000** retrieval operations to compute the mean ($\mu$) and standard deviation ($\sigma$) of the retrieval latency.

The online service's latency is simulated by **sampling from a normal distribution** $\mathcal{N}(\mu, \sigma)$.


### To start this service, please fill the following path in SpeculativeRetriever.py

encoder_name: semantic encoder used for retrieval

precomputed_random_corpus_result_dir: pre-computed retrieval results