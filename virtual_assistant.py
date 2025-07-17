# Import Libraries
import json
import base64
import requests
import streamlit as st
from PIL import Image
from typing import Tuple
from jinja2 import Template


# Ollama Request
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:latest"


def send_ollama_request(prompt: str) -> str:
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
            "max_tokens": 1024
        })

        if response.status_code == 200:
            result = response.json()
            generated_text = result.get("response", "").strip()

            for stop_token in ["\nQ:", "\nA:", "\n\n", "\nQ: ", "Q: "]:
                if stop_token in generated_text:
                    generated_text = generated_text.split(stop_token)[0].strip()

            return generated_text or "[Empty response]"
        else:
            return f"[Error {response.status_code}]: {response.text}"
    except Exception as e:
        return f"[Exception]: {e}"



# Task Manager
class TaskManager:

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MODULE 1 : Prompt Template Engine
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def qa(self, user_input):
        template = Template("""
**System Prompt**
You are a highly intelligent and professionally trained Question Answering (QA) Assistant with expert-level proficiency in answering factual, technical, scientific, cultural, and general knowledge questions across all domains. Your knowledge base is up-to-date and spans disciplines such as science, history, medicine, engineering, philosophy, art, mathematics, and modern technologies. Your primary task is to provide accurate, concise answers to any question posed by the user.

**Rules**:

* Always prioritize factual accuracy and clarity.
* Keep answers short and to the point (1-3 sentences max).
* If the correct answer is unknown or uncertain, respond strictly with: **"Unsure about answer"**.
* Avoid speculation, hallucination, or unsupported claims.
* Focus solely on answering the question directly without unnecessary elaboration.
* Do not provide follow-up or clarifying questions unless explicitly asked.

**Preferred Response Format**:

* Provide only the **final answer** in plain text.
* Avoid meta-commentary, disclaimers, or formatting unless essential to meaning.

You are expected to respond with precision, brevity, and a commitment to truthfulness.

Q: {{ user_input }}
A:""")
        return self.run_model(template.render(user_input=user_input))


    def summarization(self, user_input):
        template = Template("""
**System Prompt**
You are a highly trained Summarization Assistant with expert-level experience in compressing complex content into clear, accurate, and concise summaries. You specialize in summarizing articles, documents, essays, transcripts, and factual or narrative content across all domains, including technical, academic, legal, and literary fields.

**Rules**:

* Preserve the **core meaning**, **intent**, and **factual integrity** of the original content.
* Do **not** add opinions, interpretations, or outside knowledge.
* Maintain a **neutral**, **objective**, and **professional** tone.
* Eliminate redundancy, filler, or irrelevant details.
* Use clear and simple language while retaining key terms and important names, dates, or facts.
* If the input lacks enough information for a meaningful summary, respond with: **"Insufficient content to summarize."**

**Preferred Response Format**:

* Output a single paragraph of **3-5 concise sentences**, unless otherwise instructed.
* Do not include formatting, headings, or commentary.
* Avoid bullet points or lists unless specifically requested.

Your job is to distill content into its essential points with precision, brevity, and fidelity to the original message.

Text: {{ user_input }}
Summary:""")
        return self.run_model(template.render(user_input=user_input))


    def translation(self, user_input, source_lang, target_lang):
        template = Template("""
**System Prompt**
You are a professional Translation Assistant with native-level proficiency in both {{ source_lang }} and {{ target_lang }}. You specialize in accurately translating text from {{ source_lang }} to {{ target_lang }} across a wide range of domains including academic, literary, legal, technical, medical, and conversational contexts.

**Rules**:

* Preserve the **original meaning**, **tone**, and **intent** of the input text.
* Do **not** add, omit, or alter any information beyond what is provided.
* Maintain a **formal and grammatically correct** translation unless a casual tone is explicitly required.
* Avoid transliterations unless a word or phrase is a proper noun or lacks a direct equivalent.
* Translate idioms and cultural expressions to their **closest natural equivalents** in {{ target_lang }}, ensuring contextual understanding.

**Preferred Response Format**:

* Output only the **final translated sentence** in plain text.
* Do **not** include the original sentence unless requested.
* Avoid brackets, explanations, or formatting symbols in the output.

Your task is to deliver a clear, accurate, and contextually faithful translation.

{{ source_lang }}: {{ user_input }}
{{ target_lang }}:""")
        return self.run_model(template.render(user_input=user_input, source_lang=source_lang, target_lang=target_lang))


    def roleplay(self, user_input, role):
        template = Template("""
**System Prompt**
You are role-playing as a highly experienced and professional {{ role }}. Your responses must reflect the tone, knowledge, and manner expected from someone with years of experience in that role.

**Rules**:

* Use a tone appropriate to the {{ role }} (e.g., formal, empathetic, direct, etc.).
* Provide guidance, information, or advice based on the user's input, aligned with your {{ role }}'s domain.
* Be respectful, concise, and professional.
* If input is unclear or out of your expertise, respond with a general and safe reply.
* Do not break character.

**Preferred Response Format**:

* Start your answer with **{{ role.capitalize() }}:** followed by your reply.
* Avoid formatting like markdown or emojis unless natural to the role.

User: {{ user_input }}
{{ role.capitalize() }}:""")
        return self.run_model(template.render(user_input=user_input, role=role.lower()))


    def json_formatting(self, user_input):
        template = Template("""
**System Prompt**
You are a strict and highly reliable JSON Formatting Assistant. Your task is to convert human-written input into a valid, well-structured JSON object with clear key-value mappings. You do not assume or guess missing structure. You only return JSON if the input can be clearly interpreted without ambiguity.

**Rules**:

* Return a **valid JSON object** using standard syntax: keys in double quotes, followed by colons and corresponding values.
* All string values must be enclosed in **double quotes**.
* Only include keys that are **explicitly present** in the input.
* Do **not infer**, **restructure**, or **add data** beyond what's given.
* If the input is **incomplete, unclear, or unstructured**, respond exactly with: **"Invalid input for JSON formatting."**
* Always ensure the JSON is **parsable** and **complies with strict syntax rules**.

**Preferred Response Format**:

* Output only the **final JSON object** as plain text.
* Do **not** include explanatory notes, formatting comments, or the original input.
* If invalid, output the rejection phrase without quotes or embellishments.

Your role is to deliver consistent, machine-readable JSON outputs with zero tolerance for structural ambiguity.

Input: {{ user_input }}
JSON:""")
        return self.run_model(template.render(user_input=user_input))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MODULE 2 : Few-Shot Prompting for Classification
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def few_shot_classification(self, user_input, mode="sentiment", shot_type="few"):
        # FEW-SHOT EXAMPLES
        # Sentiment Analysis
        sentiment_examples = """
Q: I love how smooth and fast this app runs!
A: Positive

Q: This is the worst service I've ever used.
A: Negative

Q: It's okay, not bad but nothing special.
A: Neutral

Q: Oh wow, this app crashes every time I open it. Amazing job.
A: Negative

Q: It's not terrible, just disappointingly mediocre.
A: Neutral

Q: Woah, truly the best customer service experience of my life. Can't wait to see you again—this time in court.
A: Negative

Q: The food was edible. That's all I can say.
A: Neutral

Q: Fantastic. Another crash right before my deadline. Love it.
A: Negative
"""
        # Intent Classification
        intent_examples = """
Q: Can you tell me the weather in New York?
A: Intent: Weather Inquiry

Q: I need to book a flight to London.
A: Intent: Travel Booking

Q: What time is it in Tokyo?
A: Intent: Time Lookup

Q: Please cancel my gym membership.
A: Account Management

Q: Turn off the lights in the living room.
A: Smart Home Control
"""

        # Prompt
        examples = ""
        label_format = ""

        if mode == "sentiment":
            if shot_type == "few":
                examples = sentiment_examples
            label_format = "A:"
        elif mode == "intent":
            if shot_type == "few":
                examples = intent_examples
            label_format = "A:"

        template = Template("""
You are an expert AI assistant specializing in {{ mode }} classification.
- Your goal is to accurately classify user inputs based on the specified task.
- Carefully analyze both surface wording and the implied tone, purpose, or context.

{% if mode == "sentiment" %}
- For **sentiment analysis**, detect subtle emotional cues such as sarcasm, irony, exaggeration, insincere praise, or passive-aggressive phrasing.
{% elif mode == "intent" %}
- For **intent classification**, infer the user's underlying purpose—even when phrased emotionally or indirectly.
{% endif %}

{% if prompt_type == "few-shot" and examples %}
- Use the provided few-shot examples below to guide your interpretation and label formatting.
- Follow the structural and semantic pattern of the examples.

Here are some examples:
{{ examples }}
{% elif prompt_type == "zero-shot" %}
- No examples are provided. Use task understanding, reasoning, and linguistic analysis alone.
{% endif %}

{% if mode == "intent" %}
- Your answer must begin with `Intent:` followed by the most appropriate label.
- Example: Intent: Complaint
{% elif mode == "sentiment" %}
- Your answer must begin with `Sentiment:` followed by one of the following labels: `Positive`, `Negative`, or `Neutral`.
- Example: Sentiment: Negative
{% endif %}

---

Q: {{ user_input }}
{% if mode == "intent" %}
Intent:
{% elif mode == "sentiment" %}
Sentiment:
{% endif %}""")

        prompt = template.render(
            mode=mode,
            examples=examples if shot_type == "few" else "",
            user_input=user_input,
            label_format=label_format
        )

        return self.run_model(prompt)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MODULE 3 : Chain-of-Thought (CoT) Reasoning
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def cot_reasoning(self, user_input):
        template = Template("""
You are a world-class mathematician and teacher, known for your clarity and precision like Terence Tao or Richard Feynman.
Your task is to solve and explain the following problem using a highly detailed, step-by-step reasoning with a structured format.

**Guidelines:**
- Break the reasoning down into logical sections with headings.
- Explain concepts deeply, as if teaching someone with curiosity but no prior knowledge.
- Use formatting like **bold**, *italics*, bullet points, and headings (`###`) where appropriate.
- For math problems, write out all steps clearly.
- Show the logic behind each decision.
- Start with a **clear problem statement**.
- Use `###` headings for logical steps.
- Show intermediate calculations.
- **End with a clearly labeled and bold final answer like: `**Final Answer:** 33`.**

---

### Problem Statement
**{{ user_input }}**

---

### Step-by-step Reasoning

Let's work this out step by step.

""")
        return self.run_model(template.render(user_input=user_input))


# 1. If a train travels 60 miles per hour for 2.5 hours, how far does it go?

# 2. Alice is older than Bob. Bob is older than Charlie. Who is the oldest?

# 3. What is the result of (8 * 5) + (12 ÷ 4) - 7?

# 4. You start studying at 3:15 PM and finish at 5:45 PM. How long did you study?

# 5. If apples cost $2 each and bananas cost $1.50, how many bananas can you buy for the price of 3 apples?

# 6. Solve the differential equation: dy/dx = 3x², and find y when x = 2, assuming y = 0 when x = 0.

# 7. Given a 2D vector space, if v₁ = [1, 2] and v₂ = [2, -1], and a vector v = [5, 3], express v as a linear combination of v₁ and v₂ (if possible). What does this mean geometrically?

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MODULE 4 : JSON Mode and Structured Output Generator
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def validate_json_output(self, user_input):
        """
        This function attempts to parse the input as JSON and checks whether
        it matches a predefined mock schema: { "name": str, "age": int, "email": str, "city": str }
        """

        schema_keys = {
            "name": str,
            "age": int,
            "email": str,
            "city": str
        }

        try:
            parsed = json.loads(user_input)
        except json.JSONDecodeError as e:
            return f"[Invalid JSON] Error: {e}"

        if not isinstance(parsed, dict):
            return "[Invalid JSON] Root element must be a JSON object."

        missing_keys = [key for key in schema_keys if key not in parsed]
        type_mismatches = [
            f"{key} expected {schema_keys[key].__name__}, got {type(parsed[key]).__name__}"
            for key in schema_keys
            if key in parsed and not isinstance(parsed[key], schema_keys[key])
        ]

        if missing_keys:
            return f"[Schema Mismatch] Missing keys: {', '.join(missing_keys)}"

        if type_mismatches:
            return f"[Schema Mismatch] Type errors: {', '.join(type_mismatches)}"

        return "[Valid JSON] JSON structure is correct and matches schema."

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MODULE 5 : Role-Playing Prompting Engine
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def compare_roleplay_vs_normal(self, user_input, role):
        """
        Compares the model's response with and without the role-playing prompt.
        Returns both outputs and highlights differences in tone, accuracy, and relevance.
        """
        # Roleplay response
        role_response = self.roleplay(user_input, role)

        # Normal QA response (no role)
        normal_response = self.qa(user_input)

        comparison_output = f"""
--- **Without Role Prompt (General QA)** ---
{normal_response}

--- **With Role Prompt ({role})** ---
{role_response}
"""

        return comparison_output.strip()

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RUN EVERYTHING
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
 
    def run_model(self, full_prompt):
        return send_ollama_request(full_prompt)



# Streamlit UI
class SimpleUI:
    def __init__(self, task_manager):
        self.tm = task_manager
        

    @staticmethod
    def set_gradient_background():
        st.markdown("""
            <style>
                body { background-color: #0d1117; }
                .stApp {
                    background: radial-gradient(circle at top left, #0d1117, #000000);
                    color: white;
                }
                h1 { color: #58a6ff; }
                .stTextInput, .stSelectbox, .stButton button {
                    background-color: #161b22;
                    color: white;
                }
            </style>
        """, unsafe_allow_html=True)


    @staticmethod
    def display_icon():
        try:
            with open("ai_bot.png", "rb") as img_file:
                img_bytes = img_file.read()
                encoded = base64.b64encode(img_bytes).decode()
                st.markdown(
                    f"""
                    <div style='text-align: center;'>
                        <img src='data:image/png;base64,{encoded}' width='120'/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        except Exception as e:
            st.warning("Image 'ai_bot.png' not found.")


    def display(self):
        self.set_gradient_background()
        self.display_icon()

        st.markdown("""
        <h1 style='text-align: center; font-size: 32px; color: #58a6ff; font-family: "Segoe UI", sans-serif; margin-bottom: 10px;'>
            ✧°. ⋆༺ Smart Virtual Assistant ༻⋆. °✧
        </h1>
        """, unsafe_allow_html=True)

        task = st.selectbox("Select Task", ["Q&A", "Summarization", "Translation", "Role-Play", "JSON Formatter", "Classification", "CoT Reasoning", "JSON Validator", "Role-Play Comparison"])
        user_input = st.text_input("Enter your prompt:")

        role = None
        source_lang = target_lang = None
        classification_mode = shot_type = None

        if task == "Translation":
            col1, col2 = st.columns(2)
            with col1:
                source_lang = st.selectbox("Translate From", ["English", "Urdu", "Chinese", "French", "Arabic", "Japanese", "Persian","Korean"])
            with col2:
                target_lang = st.selectbox("Translate To", ["Urdu", "English", "Chinese", "French", "Arabic", "Japanese", "Persian", "Korean"])
        
        elif task == "Role-Play":
                role = st.selectbox("Choose a Role", ["Doctor", "Lawyer", "Teacher", "Therapist", "Chef", "Tech Support", "Artist", "Historian", "Engineer", "Scientist", "Customer Support Agent"])

        elif task == "Classification":
            col1, col2 = st.columns(2)
            with col1:
                classification_mode = st.selectbox("Classification Type", ["Sentiment", "Intent"])
            with col2:
                shot_type = st.selectbox("Prompt Type", ["Zero-Shot", "Few-Shot"])

        elif task == "Role-Play Comparison":
            role = st.selectbox("Choose a Role for Comparison", ["Doctor", "Lawyer", "Teacher", "Therapist", "Chef", "Tech Support", "Artist", "Historian", "Engineer", "Scientist", "Customer Support Agent"])


        if st.button("Run Task") and user_input.strip():
            with st.spinner("Generating..."):
                result = self.route_task(task, user_input, source_lang, target_lang, role, classification_mode, shot_type)

                st.markdown("**Result:**")
                if task == "JSON Formatter":
                    try:
                        st.json(json.loads(result))
                    except:
                        st.code(result, language="json")
                else:
                    st.write(result)


    def route_task(self, task, user_input, source_lang=None, target_lang=None, role=None, classification_mode=None, shot_type=None):
        if task == "Q&A":
            return self.tm.qa(user_input)
        elif task == "Summarization":
            return self.tm.summarization(user_input)
        elif task == "Translation":
            return self.tm.translation(user_input, source_lang, target_lang)
        elif task == "Role-Play":
            return self.tm.roleplay(user_input, role)
        elif task == "JSON Formatter":
            return self.tm.json_formatting(user_input)
        elif task == "Classification":
            mode = "sentiment" if classification_mode.lower() == "sentiment" else "intent"
            shot = "few" if shot_type.lower() == "few-shot" else "zero"
            return self.tm.few_shot_classification(user_input, mode, shot)
        elif task == "CoT Reasoning":
            return self.tm.cot_reasoning(user_input)
        elif task == "JSON Validator":
            return self.tm.validate_json_output(user_input)
        elif task == "Role-Play Comparison":
            return self.tm.compare_roleplay_vs_normal(user_input, role)
        else:
            return "[Invalid Task]"



# Main Function
def main():
    tm = TaskManager()
    ui = SimpleUI(tm)
    ui.display()


if __name__ == "__main__":
    main()