# Instagram Likes Prediction - Solutions Summary

Current performance: **R² = 0.17** (text embeddings only)
Goal: Improve prediction accuracy

---

## 🔥 HIGH IMPACT SOLUTIONS (Expected R² > 0.40)

### 1. **Feature Engineering with Metadata** ✅ IMPLEMENTED
**Script**: `src/training/train_likes_with_features.py`

Combine text embeddings + numerical features:
- **Influencer features**: followers, posts_count, follower_ratio
- **Post features**: comment_count, hashtag_count, is_video, caption_length

**Why it works**: Follower count is the strongest predictor of likes.

**Expected R²**: 0.40-0.60

```bash
python src/training/train_likes_with_features.py --size 10k
```

---

### 2. **Fine-Tune Transformer Models** ✅ IMPLEMENTED
**Script**: `src/training/train_transformer_finetune.py`

Train BERT/RoBERTa end-to-end on your data instead of frozen embeddings.

**Why it works**: Model learns task-specific representations.

**Expected R²**: 0.30-0.45 (text-only), 0.50+ (with features)

```bash
# Requires: pip install transformers accelerate
python src/training/train_transformer_finetune.py --size 10k --model distilbert-base-uncased --epochs 5

# Try other models:
# --model roberta-base (better quality, slower)
# --model bert-base-uncased (classic BERT)
```

---

### 3. **Predict Engagement Rate Instead** ✅ IMPLEMENTED
**Script**: `src/dataset/engagement_rate_loader.py`

Target: `engagement_rate = likes / followers` instead of absolute likes.

**Why it works**:
- 10k likes from 100k followers = 10% (good)
- 10k likes from 1M followers = 1% (poor)
- Normalizes the problem

**Expected R²**: 0.35-0.50

```python
from src.dataset.engagement_rate_loader import load_engagement_rate_data

df = load_engagement_rate_data(DATA_DIR, min_followers=100)
# Then use existing training scripts
```

---

### 4. **Multi-Modal: Text + Image Embeddings**

Use visual features from post images (you're currently ignoring them!).

**Implementation**:
```python
# Use CLIP or ResNet to get image embeddings
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# For each post:
image = Image.open(image_path)
image_inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**image_inputs)

# Combine with text embeddings:
combined_features = np.hstack([text_embeddings, image_features])
```

**Why it works**: Instagram is visual! Images drive engagement.

**Expected R²**: 0.50-0.70 (combined text+image+metadata)

---

### 5. **Use Better Embeddings**

Try OpenAI, Cohere, or Voyage embeddings instead of SentenceTransformers:

```python
# OpenAI Embeddings (1536 dims, SOTA quality)
import openai
response = openai.Embedding.create(
    model="text-embedding-3-large",
    input=texts
)
embeddings = [d['embedding'] for d in response['data']]

# Cohere Embeddings
import cohere
co = cohere.Client(api_key='YOUR_KEY')
embeddings = co.embed(texts=texts, model='embed-english-v3.0').embeddings
```

**Expected R²**: +5-10% over SentenceTransformers

---

## 🚀 MEDIUM IMPACT SOLUTIONS (Expected R² +0.05-0.15)

### 6. **Ensemble Methods**

Combine predictions from multiple models:

```python
from sklearn.ensemble import VotingRegressor

ensemble = VotingRegressor([
    ('ridge', Ridge(alpha=0.1)),
    ('rf', RandomForestRegressor(n_estimators=300)),
    ('xgb', XGBRegressor(n_estimators=300)),
])

ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
```

**Or weighted average**:
```python
y_pred_final = 0.3 * ridge_pred + 0.3 * xgb_pred + 0.4 * nn_pred
```

---

### 7. **Advanced Feature Engineering**

Add interaction features:
```python
# Follower-based features
df['likes_per_follower'] = df['likes'] / df['followers']
df['comments_per_follower'] = df['comments'] / df['followers']

# Text-based features
df['text_quality_score'] = df['caption_length'] * df['emoji_count']
df['hashtag_density'] = df['hashtag_count'] / df['caption_length']

# Temporal features (if timestamps available)
df['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5
```

---

### 8. **Two-Stage Model**

1. **Stage 1**: Predict engagement rate (likes/followers)
2. **Stage 2**: Multiply by actual follower count

```python
# Train on engagement rate
model.fit(X, engagement_rate)
predicted_rate = model.predict(X_test)

# Convert back to likes
predicted_likes = predicted_rate * follower_counts
```

---

### 9. **Add Historical Performance**

If you have multiple posts per influencer:
```python
# Average past performance
influencer_avg_likes = df.groupby('influencer_name')['likes'].transform('mean')
influencer_avg_engagement = df.groupby('influencer_name')['engagement_rate'].transform('mean')

# Add as features
df['influencer_avg_likes'] = influencer_avg_likes
df['deviation_from_avg'] = df['likes'] - influencer_avg_likes
```

---

### 10. **Topic Modeling**

Extract topics from captions:
```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Extract topics
vectorizer = CountVectorizer(max_features=1000)
X_counts = vectorizer.fit_transform(df['text'])

lda = LatentDirichletAllocation(n_components=10)
topic_features = lda.fit_transform(X_counts)

# Combine with embeddings
X_combined = np.hstack([text_embeddings, topic_features])
```

---

## 💡 ALTERNATIVE APPROACHES

### 11. **Change Problem: Classification Instead**

Instead of predicting exact likes, predict bins:
- **Low**: 0-500 likes
- **Medium**: 500-5000 likes
- **High**: 5000+ likes

**Why**: Easier problem, may be more useful for business decisions.

```python
from src.dataset.dataloader import convert_to_classification_labels
df = convert_to_classification_labels(df)
# Use classification models (already in your code)
```

---

### 12. **Outlier Handling**

Your data has huge variance (1 to 3M likes). Try:

```python
# Remove extreme outliers
Q1 = df['likes'].quantile(0.25)
Q3 = df['likes'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = df[(df['likes'] >= Q1 - 1.5*IQR) & (df['likes'] <= Q3 + 1.5*IQR)]

# Or use quantile transformation
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
y_transformed = qt.fit_transform(df[['likes']])
```

---

### 13. **Neural Architecture Search (NAS)**

Try AutoML to find optimal architecture:

```python
# Using AutoKeras
import autokeras as ak

reg = ak.StructuredDataRegressor(max_trials=20, overwrite=True)
reg.fit(X_train, y_train, epochs=50)
y_pred = reg.predict(X_test)
```

---

### 14. **External Data Enrichment**

Add context from external sources:
- **Hashtag popularity**: Use Instagram API or scrape trending hashtags
- **Seasonal trends**: Holiday periods, summer/winter
- **Influencer tier**: Micro (<100k), Mid (100k-1M), Macro (>1M)

---

### 15. **Influencer Clustering**

Group similar influencers and train separate models:

```python
from sklearn.cluster import KMeans

# Cluster by follower count, category, engagement patterns
kmeans = KMeans(n_clusters=5)
df['cluster'] = kmeans.fit_predict(df[['followers', 'posts_count', 'avg_engagement']])

# Train model per cluster
for cluster_id in range(5):
    cluster_data = df[df['cluster'] == cluster_id]
    model = train_model(cluster_data)
```

---

## 📊 RECOMMENDED IMPLEMENTATION ORDER

### Phase 1: Quick Wins (1-2 days)
1. ✅ Run `train_likes_with_features.py` (follower count is key!)
2. ✅ Try engagement rate prediction
3. Test neural networks with features

### Phase 2: Advanced Text (3-5 days)
4. Fine-tune transformer (`train_transformer_finetune.py`)
5. Try OpenAI/Cohere embeddings
6. Ensemble methods

### Phase 3: Multi-Modal (1-2 weeks)
7. Add image embeddings (CLIP)
8. Combine text + image + metadata
9. Two-stage modeling

### Phase 4: Production (if needed)
10. Outlier handling
11. Historical features
12. AutoML/NAS

---

## 🎯 EXPECTED FINAL PERFORMANCE

| Approach | Expected R² | Complexity |
|----------|-------------|------------|
| Text embeddings only | 0.15-0.20 | Low ✅ |
| Text + metadata | **0.40-0.60** | Medium ✅ |
| Fine-tuned transformer | 0.30-0.45 | Medium |
| Engagement rate | 0.35-0.50 | Low |
| Text + images + metadata | **0.60-0.75** | High |
| Full ensemble + features | **0.70-0.80** | Very High |

---

## 🚀 NEXT STEPS

**RECOMMENDED**: Start with #1 (feature engineering) - it will give you the biggest boost.

```bash
# Run this first:
python src/training/train_likes_with_features.py --size 10k

# If still not satisfied, add images:
# Implement CLIP image embeddings and combine with text+features
```

**Questions to consider**:
1. What's your target R²? (0.5 may be "good enough" for this noisy problem)
2. Do you have access to images? (Multi-modal would be huge)
3. Do you have timestamps? (Temporal features help)
4. Is this for production or research? (Affects complexity trade-offs)
