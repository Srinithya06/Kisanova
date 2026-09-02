# 🌾 Farmer Government Schemes Chatbot (CLI)

A simple AI-based command-line chatbot that helps farmers discover relevant government schemes based on their state, land, crops, and farmer category.

This project was developed during **InnovateYou Techathon 2026** to improve awareness and accessibility of government support programs for farmers.

---

## 🚀 Features

✅ Interactive chatbot in command-line
✅ Machine learning-based scheme recommendation using Nearest Neighbors  
✅ Personalized scheme recommendations
✅ Filters based on:

* State
* Land ownership
* Land area
* Crops and farming activities
* Farmer category

✅ Displays:

* Scheme benefits
* Required documents
* Official links

✅ Lightweight and easy to use
✅ Can be extended to web or mobile apps

---

## 🧠 Problem Statement

Many farmers are unaware of government schemes due to lack of access to digital platforms and complex information systems. This chatbot simplifies the process by guiding farmers step-by-step and recommending schemes suited to their needs.

---

## 💡 Solution

This chatbot collects farmer details and matches them with suitable government schemes stored in a structured dataset. It ensures that farmers receive relevant, personalized, and easy-to-understand information.

---

## 🛠️ Tech Stack

* Python
* Flask (Backend API)
* Machine Learning – Nearest Neighbors Algorithm
* JSON (for scheme database)
* CLI-based chatbot
* Rule-based + ML hybrid recommendation system

---

## 📂 Project Structure

```bash
chatbot_kisan/
│
├── chatbot.py        # Main chatbot logic  
├── main.py           # Entry point  
├── schemes.json      # Government schemes database  
├── requirements.txt  # Dependencies  
└── README.md         # Documentation  
```

---

## ⚙️ Installation & Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/harshallogade/farmer-schemes-chatbot.git
```

### Step 2: Go to project folder

```bash
cd farmer-schemes-chatbot
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the chatbot

```bash
python chatbot.py
```

---

## 🧪 How it Works

1. The chatbot asks for:

   * State
   * Land details
   * Crops or activities
   * Farmer type
2. It processes the input.
## 🤖 Machine Learning Approach

This chatbot uses a hybrid approach combining rule-based filtering and machine learning.

The **Nearest Neighbors algorithm** is used to recommend government schemes by finding similarity between the farmer’s profile and existing scheme eligibility conditions. This improves personalization and allows scalable recommendations.

Flask is used to structure the backend, making the system ready for integration with web or mobile applications in the future.

3. Matches it with the schemes dataset.
4. Displays personalized scheme recommendations.

---

## 📈 Future Improvements

* Web application version
* Mobile app
* Multilingual support (Hindi, Marathi, etc.)
* Voice assistant integration
* AI and LLM-based smart recommendations
* Live government API integration


---

## 🎯 Impact

This project can:

* Improve awareness of welfare schemes
* Help small and marginal farmers
* Support digital agriculture initiatives
* Bridge the gap between farmers and government support

---

## 🏆 Hackathon

This project was developed during **InnovateYou Techathon 2026**.

---

## 👨‍💻 Team

Developed by:

* **Harshal** and team (3 members)

---

## 🤝 Contribution

Contributions are welcome!
Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.
