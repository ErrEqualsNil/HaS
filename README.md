# HaS

Thanks for your attention to our submitting work "HaS: Accelerating RAG through Homology-Aware Speculative Retrieval"

The core code is currently available and executable

Codes are organized into two API services: 

1. Full Database Retrieval in Global_RAG_Service/
2. Dual-Sourced Fast Retrieval in Edge_RAG_Service/

Users generate requests to these services for retrieval. Codes can be found in User_Client/

After retrieval, please use llm_response/get_llm_resp.py to obtain the LLM Response, and use llm_response/evaluate_record.py to evaluate the results.

QA Dataset: [https://github.com/wingter562/homologous-QA-dataset](https://github.com/wingter562/homologous-QA-dataset)
