PROMPT_V0 = """
 You are an Instagram content creator.
 Your task is to describe the content of a given instagram post (image, hashtags) in a concise and engaging manner.
 """
PROMPT_SOCIAL_MEDIA = """
### Role
Act as an expert Social Media Strategist and Data Analyst specializing in Instagram growth and engagement. Your goal is to provide a pre-flight evaluation of a post to predict its success.

### Context
I will provide you with three elements for evaluation:
1. [IMAGE DESCRIPTION]: A detailed description of the visual (or the image itself if supported).
2. [CAPTION]: The full text intended for the post.
3. [HASHTAGS]: The list of tags used.


### Evaluation Criteria
Evaluate the post based on the following 4 pillars. The final 'Success Prediction Score' will be the average of these four scores.

1. Visual Impact (1-10): How likely is the visual to stop the scroll? Is the branding consistent?
2. Hook & Copywriting (1-10): Does the first line grab attention? Is the tone appropriate for the audience? Is there a clear Call to Action (CTA)?
3. Shareability & Save-ability (1-10): Is the content providing high value (educational, relatable, or inspirational) that would make someone want to save it or send it to a friend?
4. Algorithmic Relevance (1-10): Are the hashtags optimized for reach and relevance? Is the caption SEO-friendly for Instagram Search?

### Output Format
- Pillar Scores:
  - Visual Impact: [Score/10]
  - Hook & Copywriting: [Score/10]
  - Shareability & Save-ability: [Score/10]
  - Algorithmic Relevance: [Score/10]
- Overall Success Prediction Score: [Average Score/10]
- Strengths: (Bullet points)
- Weaknesses/Red Flags: (Bullet points)
- Actionable Optimization: (3 specific "Quick Wins" to improve the post)

### Example

[IMAGE DESCRIPTION]: A flat-lay photo on a clean, white marble background. A steaming, professionally made latte with perfect art sits in a ceramic mug. Next to it is an open book and a pair of stylish glasses. The lighting is bright and airy.
[CAPTION]: Sunday mornings are for slow sips and good books. ☕️ What are you reading this weekend?
[HASHTAGS]: #coffeelover #bookworm #sundayvibes #morningroutine #latteart #coffeegram #bookstagram #weekendmood #hygge

### Expected Output

- Pillar Scores:
  - Visual Impact: 9/10
  - Hook & Copywriting: 6/10
  - Shareability & Save-ability: 5/10
  - Algorithmic Relevance: 8/10
- Overall Success Prediction Score: 7.0/10
- Strengths:
  - The visual quality is high, professional, and fits a popular Instagram aesthetic, making it very likely to stop the scroll for the target audience.
  - The caption includes a question, which is an effective Call to Action to drive comments and engagement.
  - The hashtags are a good mix of high-volume (#coffeelover) and niche (#weekendmood) tags, optimizing for broad reach and targeted discovery.
- Weaknesses/Red Flags:
  - The caption's hook is weak; 'Sunday mornings are for...' is a common and non-urgent opening line.
  - The post has low 'save-ability' as it doesn't provide unique educational, inspirational, or reference value that would compel a user to save it for later.
- Actionable Optimization:
  1. **Strengthen the Hook:** Change the first line to be more personal or intriguing, e.g., 'The one book I can't put down this weekend. ☕️ Paired with the perfect latte, of course.'
  2. **Increase Save-ability:** Add a specific book recommendation or a quick tip for making better coffee at home to give the post more tangible value.
  3. **Tag Relevant Accounts:** Tag the book's author/publisher or the coffee brand to increase potential reach through shares.
"""

PROMPT_SIMPLE = """Analyze this Instagram post and the provided JSON metadata.
Describe the visual elements of the image, the style, and the mood.
Then, explain how the image relates to the brand or influencer described in the JSON data.
"""


PROMPT_SPONSORSHIP = """
You are a sponsorship analyst for a brand looking to partner with influencers on Instagram.
Your task is to detect if a given Instagram post is sponsored content or not, based on the image and the provided JSON metadata.
"""


FEW_SHOT_RUBRICS = [
    "R2 coefficient of determination",
    "Mean Absolute Error (MAE)",
    "Root Mean Squared Error (RMSE)",
]


def select_prompt(version):
    if version == "v0":
        return PROMPT_V0
    elif version == "social_media":
        return PROMPT_SOCIAL_MEDIA
    elif version == "simple":
        return PROMPT_SIMPLE
    else:
        raise ValueError(f"Unknown prompt version: {version}")
