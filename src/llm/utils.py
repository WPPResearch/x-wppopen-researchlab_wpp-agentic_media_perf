import os

from google import genai


from src.config import USE_VERTEX, PROJECT_ID, REGION, API_KEY


def initialize_gemini_client():
    # If API_KEY is set, we might be using AI Studio. If PROJECT_ID is set, maybe Vertex.
    # The original script prioritized Vertex if USE_VERTEX is true.

    if USE_VERTEX and PROJECT_ID and REGION:
        print(
            f"Initializing Gemini Client with Vertex AI (Project: {PROJECT_ID}, Region: {REGION})"
        )
        return genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
    elif API_KEY:
        print("Initializing Gemini Client with API Key")
        os.environ["GEMINI_API_KEY"] = API_KEY
        return genai.Client(api_key=API_KEY)
    else:
        print("Warning: No API Key or Project ID found. Initialization might fail.")
        return genai.Client()
