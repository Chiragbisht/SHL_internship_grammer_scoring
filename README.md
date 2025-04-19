📚 Grammar Assessment Model
🔍 Overview
The Grammar Assessment Model is an automated system for evaluating the grammar quality of spoken English from audio recordings.

It integrates speech-to-text technology and advanced language models to deliver objective and consistent grammar scores on a 1–5 scale.

⚙️ Key Components
🎙️ Audio Transcription Service
Uses AssemblyAI API to convert audio speech to text

Supported formats: .wav, .mp3, .m4a, .aac

Includes robust error handling for failed or incomplete transcriptions

✍️ Grammar Assessment Model
Powered by Google Gemini and AssemblyAI speech-to-text API for natural language evaluation

Scoring Scale:

pgsql
Copy
Edit
1.0 – Very poor grammar with many errors  
1.5 – Poor grammar with significant errors  
2.0 – Below average grammar with several errors  
2.5 – Slightly below average grammar with some errors  
3.0 – Average grammar with occasional errors  
3.5 – Above average grammar with few errors  
4.0 – Good grammar with minimal errors  
4.5 – Very good grammar with almost no errors  
5.0 – Excellent grammar with no errors  
Utilizes example-based scoring for consistency

🧩 Pipeline Architecture
📂 Audio File Processing
Scans and validates audio files

Ensures format compatibility and file accessibility

🔁 Transcription Processing
Uploads audio to AssemblyAI

Monitors transcription status

🧠 Grammar Assessment
Sends transcription to Gemini model

Matches against reference examples

Assigns final score

💾 Results Management
Saves scores and logs to .csv files

Computes metrics like RMSE and MAE

📊 Score Distribution

Score Range	Approx. Share
5.0 (Excellent)	~45%
4.0–4.9	~35%
3.0–3.9	~15%
1.0–2.9	~5%
