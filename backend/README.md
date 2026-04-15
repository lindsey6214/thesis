# EssayLens — Backend Setup

## Files you need

```
backend/
  app.py                          ← Flask API  (this folder)
  requirements.txt
  text_classification_model.keras ← copy from your Google Drive / Colab output
  tfidf_tokenizer.pkl             ← copy from your Google Drive / Colab output
index.html                        ← the frontend (one level up)
```

## 1 — Save your model files from Colab

Add these two lines at the end of your Colab notebook (after training):

```python
# Save TF-IDF tokenizer
import pickle
with open('/content/drive/MyDrive/Honors_Thesis/tfidf_tokenizer.pkl', 'wb') as f:
    pickle.dump(tfidf_tokenizer, f)

# Save Keras model  (you likely already do this)
model.save('/content/drive/MyDrive/Honors_Thesis/text_classification_model.keras')
```

Then download both files from Google Drive to your computer and place them
**inside the `backend/` folder** next to `app.py`.

## 2 — Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

## 3 — Run the server

```bash
python app.py
```

You should see:
```
[✓] Loaded tokenizer  → tfidf_tokenizer.pkl
[✓] Loaded Keras model → text_classification_model.keras
 * Running on http://0.0.0.0:5000
```

## 4 — Open the frontend

Open `index.html` in your browser (or serve it with `python -m http.server 8080`).
The frontend already points to `http://localhost:5000/predict`.

## API reference

### POST /predict
```json
{ "text": "paste essay here..." }
```

Response:
```json
{
  "label": 1,
  "verdict": "AI-Generated",
  "ai_probability": 0.8723,
  "confidence": 0.8723,
  "signals": {
    "word_count": 312,
    "vocab_richness": 0.61,
    "avg_sentence_len": 22.4,
    "punct_density": 1.8,
    "bigram_diversity": 0.94,
    "unique_words": 190,
    "bigram_count": 281
  }
}
```

### GET /health
Returns `{ "status": "ok", "model_loaded": true, "tokenizer_loaded": true }`
