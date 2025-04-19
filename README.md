📚 Grammar Assessment Model
🔍 Overview
An automated system for evaluating spoken English grammar from audio recordings. Combines AssemblyAI for transcription and Google Gemini for consistent scoring on a 1–5 scale.

⚙️ Key Components
🎙️ Audio Transcription Service
Converts speech to text using AssemblyAI API

Supported formats: WAV, MP3, M4A, AAC

Built-in error handling for failed/incomplete transcriptions

✍️ Grammar Assessment Model
Uses Google Gemini for natural language evaluation

Scoring Scale:

scss
Copy
Edit
1.0  – Very poor grammar
1.5  – Poor grammar
2.0  – Below average
2.5  – Slightly below average
3.0  – Average
3.5  – Above average
4.0  – Good
4.5  – Very good
5.0  – Excellent
Example-based scoring for better consistency

🧩 Pipeline Architecture
📂 Audio File Processing
Validates audio files and format compatibility

🔁 Transcription Processing
Uploads and monitors status via AssemblyAI

🧠 Grammar Assessment
Sends transcription to Gemini

Matches against reference examples

Assigns final score

📊 Results Management
Stores scores/logs in CSV

Computes metrics like RMSE and MAE

📈 Score Distribution

Score Range	Approx. Share
5.0	~45%
4.0–4.9	~35%
3.0–3.9	~15%
1.0–2.9	~5%
⚡ Highlights
✅ Accurate transcription via AssemblyAI

✅ Powerful grammar scoring via Gemini-2.0-flash

✅ Consistent logic using reference benchmarks

✅ Reproducible pipeline from input to output

📊 Performance Analysis
Based on assessment_results.csv with 204 audio samples:

📌 Dataset Summary
Total files processed: 204

Score range: 1.0 to 5.0

Mean score: 4.15

Median score: 4.5

Standard deviation: 0.98

📐 Metrics
RMSE: 0.3655

MAE: 0.2672

⚖️ The model typically predicts within ±0.37 points of the true score and maintains high reliability—suitable for both educational and professional applications.
