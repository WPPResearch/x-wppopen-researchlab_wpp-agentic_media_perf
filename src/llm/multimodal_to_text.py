from google.genai import types


def llm_post_to_text(
    client,
    images: list,
    prompt: str,
    model_name: str = "gemini-2.5-flash",
    temperature: float = 0.0,
):
    """
    Classifies an icon image into one of the provided labels using Gemini.
    """

    config = types.GenerateContentConfig(
        temperature=temperature,  # Low temperature for deterministic output
    )

    contents = []
    for image in images:
        contents.append(image)
    contents.append(prompt)
    try:
        response = client.models.generate_content(
            model=model_name, contents=contents, config=config
        )
        text_response = response.text
        # Clean up potential markdown code blocks if the model ignores the instruction
        text_response = text_response.replace("```json", "").replace("```", "").strip()
    except Exception as e:
        print(f"Error during prediction: {e}")
        text_response = None
    return text_response
