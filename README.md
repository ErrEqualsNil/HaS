# HaS

Thanks for your attention to our submitting work "Re-identify to Get It Right: Accelerating RAG through Homology-Aware Speculative Retrieval"

The core code is currently available and executable

Codes are organized into two API services: 

1. Full Corpus Retrieval in Global_RAG_Service folder
2. Speculative Retrieval in Edge_RAG_Service folder

Users generate requests to these services for retrieval. Codes can be found in User_Client folder.

After retrieval, please use llm_response/get_llm_resp.py to obtain LLM Response, and use llm_response/evaluate_record.py to evaluate the results.
