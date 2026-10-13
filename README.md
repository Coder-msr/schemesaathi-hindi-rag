# SchemeSaathi — Hindi/Hinglish/English Government Scheme RAG Prototype

A research prototype for answering questions from official government scheme guideline PDFs. It retrieves document passages, shows source filename and PDF page number, and abstains when no evidence is found.

## Included functionality
- Hindi, Roman Hindi/Hinglish, and English questions
- Attempts to download official PMAY-U 2.0 Hindi and English guideline PDFs on first launch
- Manual PDF upload for other official schemes or if hosting blocks automatic downloads
- Multilingual semantic retrieval (Sentence Transformers) with keyword fallback
- Source filename, PDF page number, passage text and retrieval score
- Explicit research-prototype disclaimer; not an official government service

## Official PMAY-U 2.0 documents
- Hindi: https://pmay-urban.gov.in/uploads/guidelines/Operational-Guidelines-of-PMAY-U-2-Hindi.pdf
- English: https://pmay-urban.gov.in/uploads/guidelines/Operational-Guidelines-of-PMAY-U-2.pdf
- Ministry listing: https://pmay-urban.gov.in/guideline

The PDFs are downloaded at app startup when the hosting environment allows outbound HTTP. If not, download the PDFs from the official links above and upload them in the app sidebar. These are PMAY-U 2.0 documents only; other schemes need their own official source PDFs.

## Run locally
1. Install Python 3.10–3.12.
2. Extract this ZIP.
3. Open a terminal in the `SchemeSaathi` folder.
4. Run `pip install -r requirements.txt`.
5. Run `streamlit run app.py`.

The first semantic search may download the multilingual embedding model and take a few minutes. If that fails, disable semantic retrieval in the sidebar to use keyword matching.

## Deploy online (GitHub + Streamlit Community Cloud)
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this README at the repository root. Do not upload private information.
2. Visit https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select your repository/branch and `app.py`, then click **Deploy**.
4. When the deployment finishes, copy the generated `*.streamlit.app` URL.

Official deployment instructions: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

I cannot publish it to your account or create a real live URL without your authenticated GitHub/hosting action. Never send passwords or access tokens in chat.

## Research integrity and limitations
- This prototype retrieves passages; it does not provide a fully generative LLM answer.
- Semantic retrieval depends on downloading a multilingual embedding model.
- PDF text extraction may fail for scanned documents requiring OCR.
- Retrieval scores are not factual accuracy scores. Evaluate against a manually verified question/answer benchmark before reporting accuracy.
- Check official guideline updates and amendments before using scheme information for real applications.
