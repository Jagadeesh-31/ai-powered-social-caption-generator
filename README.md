# AI-Powered Multi-Platform Social Media Caption Generator

This project runs image captioning locally using the BLIP model and automatically translates the image context into highly engaging, platform-specific social media captions (for Instagram, X/Twitter, Facebook, LinkedIn, and Pinterest) using the Google Gemini API.

---

## Local Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jagadeesh-31/ai-powered-social-caption-generator.git
   cd ai-powered-social-caption-generator
   ```

2. **Initialize Virtual Environment:**
   * **Windows:**
     ```powershell
     python -m venv venv
     .\venv\Scripts\activate
     ```
   * **Mac/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Your API Key:**
   * **Windows (PowerShell):** `$env:GEMINI_API_KEY="your-api-key"`
   * **Mac/Linux:** `export GEMINI_API_KEY="your-api-key"`

5. **Run the Streamlit Web Application:**
   ```bash
   streamlit run 7_streamlit_social_demo.py
   ```

---

## Deployment to Render

To deploy this Streamlit app to Render:

1. Log in to **[Render.com](https://render.com/)** and click **New > Web Service**.
2. Connect your GitHub repository: `ai-powered-social-caption-generator`.
3. Configure the service:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `streamlit run 7_streamlit_social_demo.py --server.port $PORT --server.address 0.0.0.0`
   * **Instance Type:** Choose a paid tier (e.g. **Starter** or **Standard** with at least 2GB RAM).
     > [!WARNING]
     > The free tier of Render has a **512MB RAM limit**. Loading the 1GB PyTorch-based BLIP model will exceed this limit and cause a container crash (Out Of Memory / OOM).
4. Add **Environment Variables** in the Render settings:
   * **Key:** `GEMINI_API_KEY`
   * **Value:** `your-actual-api-key`
