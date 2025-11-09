# Jarvis-AI
Jarvis — Local Document Assistant with Authentication is a Streamlit-based AI tool that allows users to securely upload and interact with their own documents locally without any external APIs or cloud dependencies. The system supports PDF, DOCX, and TXT formats, enabling users to upload a file, generate a concise five-line summary, and ask context-based questions directly from the uploaded document using a built-in TF-IDF and cosine similarity search engine. It includes a secure login and signup system using email and password authentication stored in SQLite with PBKDF2 password hashing for safety. Jarvis also supports voice input for speech-to-text queries, text-to-speech responses, and lets users export their chat history to PDF or Word format. The app offers a modern, minimalist UI with sidebar history, new chat reset, and a fully offline workflow — making it ideal for research, education, and document analysis use cases.
| Component           | Technology                                     |
| ------------------- | ---------------------------------------------- |
| **Frontend**        | Streamlit                                      |
| **Language**        | Python 3                                       |
| **Storage**         | SQLite (for users), in-memory (for documents)  |
| **Text Processing** | TF-IDF, Regex, PyPDF2, python-docx             |
| **Voice & TTS**     | `sounddevice`, `speech_recognition`, `pyttsx3` |
| **Security**        | PBKDF2 password hashing                        |
| **Export**          | FPDF (for PDF), python-docx (for DOCX)         |

📁 Jarvis-Local-Assistant/
│
├── app_with_uploader_pro_with_auth.py    # Main Streamlit app
├── users.db                              # SQLite database for user accounts
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
└── .env (optional)                       # Environment variables (if needed)

