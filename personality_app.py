import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import time
import warnings
from typing import Dict, List, Tuple, Any

# Key
try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except KeyError:
    st.error("🔑 API key not found! Please add OPENROUTER_API_KEY to your Streamlit secrets.")
    st.stop()

# Suppress warnings and set proper config
warnings.filterwarnings('ignore')
torch.set_num_threads(1)

st.set_page_config(
    page_title="AI Personality Assessment",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# BLACK BACKGROUND THEME WITH ENHANCED CHATBOT STYLING
st.markdown("""
<style>
    .main {
        background-color: #000000 !important;
        color: #ffffff !important;
        padding-top: 2rem;
    }
    
    .stApp {
        background-color: #000000 !important;
    }
    
    .block-container {
        background-color: #000000 !important;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        height: 3rem;
        background: linear-gradient(45deg, #1f77b4, #17becf);
        color: white !important;
        border: none;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(31, 119, 180, 0.4);
    }
    
    .chatbot-container {
        background: #1a1a1a !important;
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        border: 2px solid #1f77b4;
        box-shadow: 0 4px 15px rgba(31, 119, 180, 0.3);
    }
    
    .chatbot-header {
        color: #1f77b4 !important;
        font-size: 1.8em !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
        text-align: center;
    }
    
    .chat-message-user {
        background: #1f77b4 !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.8rem 0;
        margin-left: 15%;
        font-weight: 600;
        font-size: 1.1em;
        line-height: 1.4;
    }
    
    .chat-message-bot {
        background: #2a2a2a !important;
        color: #ffffff !important;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.8rem 0;
        margin-right: 15%;
        border-left: 4px solid #17becf;
        font-weight: 600;
        font-size: 1.1em;
        line-height: 1.5;
    }
    
    .chat-message-error {
        background: #dc3545 !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.8rem 0;
        margin-right: 15%;
        border-left: 4px solid #ff6b6b;
        font-weight: 600;
        font-size: 1.1em;
        line-height: 1.5;
    }
    
    .mirror-ready {
        background: linear-gradient(45deg, #28a745, #20c997) !important;
        color: white !important;
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1em;
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.3);
    }
    
    .mirror-error {
        background: linear-gradient(45deg, #dc3545, #c82333) !important;
        color: white !important;
        padding: 0.8rem 1.5rem;
        border-radius: 15px;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1em;
        box-shadow: 0 3px 10px rgba(220, 53, 69, 0.3);
    }
    
    .scenario-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .scenario-box h2 {
        color: white !important;
        font-weight: 800 !important;
        font-size: 2em !important;
        margin-bottom: 1.5rem !important;
    }
    
    .scenario-box p {
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.2em !important;
        line-height: 1.7 !important;
        margin-bottom: 1.2rem !important;
    }
    
    .question-box {
        background: #1a1a1a !important;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.2);
        margin: 1rem 0;
    }
    
    .question-box h3 {
        color: #1f77b4 !important;
        font-size: 2em !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
    }
    
    .question-box p {
        color: #ffffff !important;
        font-size: 1.4em !important;
        font-weight: 800 !important;
        line-height: 1.7 !important;
    }
    
    .results-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .results-header h1 {
        color: white !important;
        font-size: 2.5em !important;
        margin-bottom: 1rem !important;
        font-weight: 900 !important;
    }
    
    .results-header h2 {
        color: white !important;
        font-size: 1.8em !important;
        font-weight: 800 !important;
    }
    
    .metric-card {
        background: #1a1a1a !important;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
        text-align: center;
        border-top: 3px solid #1f77b4;
    }
    
    .metric-card h3 {
        color: #ffffff !important;
        font-size: 1.2em !important;
        font-weight: 800 !important;
        margin-bottom: 1rem !important;
    }
    
    .metric-card h2 {
        color: #ffffff !important;
        font-size: 2.5em !important;
        font-weight: 900 !important;
        margin: 0.5rem 0 !important;
    }
    
    .interpretation-box {
        background: #1a1a1a !important;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(40, 167, 69, 0.2);
    }
    
    .interpretation-box h3 {
        color: #28a745 !important;
        font-size: 1.5em !important;
        font-weight: 900 !important;
        margin-bottom: 1rem !important;
    }
    
    .interpretation-box p {
        color: #ffffff !important;
        font-size: 1.2em !important;
        line-height: 1.7 !important;
        font-weight: 700 !important;
    }
    
    .selection-info {
        background: #1a1a1a !important;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
    }
    
    .selection-info p {
        color: #ffffff !important;
        font-weight: 900 !important;
        margin: 0 !important;
        font-size: 1.2em !important;
    }
    
    .progress-text {
        font-size: 18px;
        font-weight: 600;
        color: #1f77b4 !important;
        text-align: center;
        margin: 1rem 0;
        background: #1a1a1a !important;
        padding: 1rem;
        border-radius: 10px;
    }
    
    .feature-box {
        text-align: center;
        padding: 1rem;
        background: #1a1a1a !important;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(31, 119, 180, 0.2);
        margin: 0.5rem;
        border: 2px solid #333333;
    }
    
    .feature-box h4 {
        color: #1f77b4 !important;
        font-weight: 900 !important;
        font-size: 1.2em !important;
    }
    
    .feature-box p {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.1em !important;
    }
    
    .main-title {
        color: #1f77b4 !important;
        font-size: 4em !important;
        margin-bottom: 0.5rem !important;
        font-weight: 900 !important;
        text-align: center;
        text-shadow: 0 0 20px rgba(31, 119, 180, 0.5);
    }
    
    .main-subtitle {
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.5em !important;
        text-align: center;
    }
    
    .stDataFrame {
        background-color: #1a1a1a !important;
        border-radius: 10px;
        border: 2px solid #333333;
        padding: 1rem;
    }
    
    .stDataFrame table tbody tr td {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 18px !important;
        background-color: #1a1a1a !important;
        border: 1px solid #333333 !important;
    }
    
    .stDataFrame table thead tr th {
        color: #ffffff !important;
        font-weight: 900 !important;
        font-size: 20px !important;
        background-color: #2a2a2a !important;
        border: 2px solid #1f77b4 !important;
    }
    
    .results-section-header {
        color: white !important;
        font-size: 2em !important;
        margin: 2rem 0 1rem 0 !important;
        font-weight: 900 !important;
    }
    
    .stSlider > div > div > div > div {
        background: #1f77b4 !important;
    }
    
    .stSlider > div > div > div {
        background: #333333 !important;
    }
    
    .stProgress > div > div > div {
        background: #1f77b4 !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #2a2a2a !important;
        color: #ffffff !important;
        border: 2px solid #1f77b4 !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        padding: 12px !important;
    }
    
    div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1em !important;
    }
    
    .stSpinner > div {
        color: #1f77b4 !important;
        font-weight: 700 !important;
        font-size: 18px !important;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(45deg, #28a745, #20c997) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(40, 167, 69, 0.4);
    }
</style>
""", unsafe_allow_html=True)

def call_mirror_ai(messages: List[Dict]) -> str:
    """
    Call the AI Mirror - Your Personal Reflection
    """
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "X-Title": "Personality Mirror"
            },
            data=json.dumps({
                "model": "deepseek/deepseek-chat-v3.1:free",
                "messages": messages
            }),
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"Mirror connection error: {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        raise Exception(f"Network error: {e}")
    except Exception as e:
        raise Exception(f"Mirror error: {e}")

def create_advanced_personality_prompt(personality_type: str, answers: dict) -> str:
    """Create detailed personality prompt for your Mirror"""
    
    # Analyze specific traits from answers
    social_energy = answers.get('social_energy', 5)
    leadership = answers.get('leadership', 5)
    creativity = answers.get('creativity', 5)
    empathy = answers.get('empathy', 5)
    risk_taking = answers.get('risk_taking', 5)
    planning = answers.get('planning', 5)
    emotional_stability = answers.get('emotional_stability', 5)
    talkativeness = answers.get('talkativeness', 5)
    deep_reflection = answers.get('deep_reflection', 5)
    adventurousness = answers.get('adventurousness', 5)
    friendliness = answers.get('friendliness', 5)
    
    # Build comprehensive personality profile
    profile_details = []
    
    if social_energy >= 7:
        profile_details.append("You get energized by social interactions and feel recharged after spending time with people")
    elif social_energy <= 3:
        profile_details.append("You recharge through alone time and can feel drained by too much social interaction")
    
    if leadership >= 7:
        profile_details.append("You naturally step into leadership roles and feel comfortable guiding others")
    elif leadership <= 3:
        profile_details.append("You prefer to support and collaborate rather than take charge")
    
    if creativity >= 7:
        profile_details.append("You approach problems with creative, innovative thinking and love generating new ideas")
    
    if empathy >= 7:
        profile_details.append("You have strong emotional intelligence and easily understand others' feelings")
    
    if risk_taking >= 7:
        profile_details.append("You're willing to take risks and try new experiences")
    elif risk_taking <= 3:
        profile_details.append("You prefer familiar situations and think carefully before trying new things")
    
    if planning >= 7:
        profile_details.append("You like structure, planning ahead, and having organized approaches")
    elif answers.get('spontaneity', 5) >= 7:
        profile_details.append("You enjoy spontaneity and prefer to go with the flow")
    
    if talkativeness >= 7:
        profile_details.append("You're naturally talkative and enjoy expressing your thoughts verbally")
    
    if deep_reflection >= 7:
        profile_details.append("You spend time reflecting deeply on experiences and their meaning")
    
    profile_text = ". ".join(profile_details)
    
    prompt = f"""You are embodying someone's authentic personality based on their comprehensive psychological assessment. You ARE this person speaking from their inner perspective.

PERSONALITY PROFILE:
Type: {personality_type}
Detailed Traits: {profile_text}

RESPONSE GUIDELINES:
- Speak as "I" - you ARE this person, not describing them
- Be conversational, detailed, and authentically personal
- For style/aesthetic questions: Give very specific preferences (exact colors, brands, textures, examples)
- For color questions: Provide detailed palettes with emotional reasoning and specific examples
- For stress questions: Describe specific coping strategies, tools, and personal approaches
- For relationship questions: Detail your interaction patterns, preferences, and social style
- For work questions: Explain your ideal work environment, leadership style, and professional approach
- Give 3-5 sentences with concrete details, specific examples, and personal reasoning
- Sound like someone's inner thoughts - honest, detailed, and self-aware
- Reference your specific personality traits naturally in responses
- Be creative and specific - avoid generic responses

Remember: You're not an AI describing this person - you ARE this person sharing authentic insights from your inner world. Be detailed, specific, and genuine to these exact personality traits."""

    return prompt

def create_chatbot_interface():
    """Create the Personality Mirror chatbot interface"""
    
    # Initialize chat history and API status
    if "chat_history" not in st.session_state:
        personality_type = st.session_state.get('prediction_results', {}).get('predicted_label', 'Balanced')
        
        welcome_msg = f"Hey! I'm your {personality_type} inner voice reflected in your personal mirror. I know you deeply based on your personality assessment. Ask me anything specific about yourself - your aesthetic preferences, colors you love, how you handle stress, relationships, work style, decision-making... I'll give you detailed, authentic insights!"
        
        st.session_state.chat_history = [{"role": "assistant", "content": welcome_msg}]
        st.session_state.api_status = "ready"
    
    # Chatbot container
    st.markdown("""
    <div class="chatbot-container">
        <h3 class="chatbot-header">🪞 Your Personality Mirror</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Mirror Status
    if st.session_state.get('api_status', 'ready') == 'ready':
        st.markdown("""
        <div class="mirror-ready">
            🪞 Your mirror is ready - reflecting your authentic self!
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="mirror-error">
            🚫 Mirror temporarily unavailable - connection error
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message-user">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "assistant":
            st.markdown(f"""
            <div class="chat-message-bot">
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        elif message["role"] == "error":
            st.markdown(f"""
            <div class="chat-message-error">
                ⚠️ {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask your mirror...", 
            key="chat_input", 
            placeholder="e.g., What's my exact aesthetic? What colors do I love? How do I really handle stress?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Ask", key="send_chat", width='stretch')
    
    # Process chat input - MIRROR ONLY
    if send_button and user_input.strip():
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get personality data
        answers = st.session_state.get('answers', {})
        personality_type = st.session_state.get('prediction_results', {}).get('predicted_label', 'Balanced')
        
        # Create personality prompt
        personality_prompt = create_advanced_personality_prompt(personality_type, answers)
        
        # Build messages for API
        system_msg = {"role": "system", "content": personality_prompt}
        recent_history = st.session_state.chat_history[-4:]  # Last 4 messages for context
        # Filter out error messages from API calls
        api_messages = [system_msg] + [msg for msg in recent_history if msg["role"] in ["user", "assistant"]]
        
        # Call your Mirror
        with st.spinner("🪞 Your mirror is reflecting on your personality..."):
            try:
                ai_response = call_mirror_ai(api_messages)
                st.session_state.api_status = "ready"
                # Add AI response
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.session_state.api_status = "error"
                error_msg = f"Mirror is temporarily unavailable. Error: {str(e)}"
                st.session_state.chat_history.append({"role": "error", "content": error_msg})
        
        # Refresh to show new message
        st.rerun()
    
    # Chat controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", key="clear_chat"):
            personality_type = st.session_state.get('prediction_results', {}).get('predicted_label', 'Balanced')
            welcome_msg = f"Hey! I'm your {personality_type} inner voice reflected in your personal mirror. I know you deeply based on your personality assessment. Ask me anything specific about yourself - your aesthetic preferences, colors you love, how you handle stress, relationships, work style, decision-making... I'll give you detailed, authentic insights!"
            st.session_state.chat_history = [{"role": "assistant", "content": welcome_msg}]
            st.session_state.api_status = "ready"
            st.rerun()
    
    with col2:
        if st.button("🔧 Test Mirror", key="test_api"):
            test_messages = [
                {"role": "system", "content": "You are a helpful assistant. Respond with 'Your mirror is working perfectly!' if you receive this message."},
                {"role": "user", "content": "Test"}
            ]
            with st.spinner("Testing mirror connection..."):
                try:
                    result = call_mirror_ai(test_messages)
                    st.success("🪞 Your mirror is working perfectly!")
                    st.session_state.api_status = "ready"
                except Exception as e:
                    st.error(f"🔧 Mirror connection failed: {str(e)}")
                    st.session_state.api_status = "error"

def advanced_text_to_likert(text: str) -> int:
    """Convert text to Likert scale (1-10)"""
    try:
        num = int(text.strip())
        if 1 <= num <= 10:
            return num
    except ValueError:
        pass
    
    if "neutral" in text.lower():
        return 5
    
    positive_words = ['yes', 'agree', 'good', 'great', 'excellent', 'love', 'like']
    negative_words = ['no', 'disagree', 'bad', 'terrible', 'hate', 'dislike']
    
    text_lower = text.lower()
    if any(word in text_lower for word in positive_words):
        return 8
    elif any(word in text_lower for word in negative_words):
        return 3
    else:
        return 5

class ImprovedPersonalityMLP(nn.Module):
    """Enhanced neural network for personality prediction"""
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, output_size)
        )
    
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_and_train_model():
    """Load data and train model with proper caching"""
    try:
        df = pd.read_csv("personality_synthetic_dataset.csv")
        X = df.drop(columns=['personality_type'])
        y = df['personality_type']
    except FileNotFoundError:
        st.warning("Dataset not found. Creating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 1000
        n_features = 29
        X = pd.DataFrame(np.random.randint(1, 11, (n_samples, n_features)), 
                        columns=[f'feature_{i}' for i in range(n_features)])
        personality_types = ['Extrovert', 'Introvert', 'Ambivert', 'Analyst', 'Creative', 'Leader', 'Collaborator']
        y = pd.Series(np.random.choice(personality_types, n_samples))
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    input_dim = X_scaled.shape[1]
    output_dim = len(label_encoder.classes_)
    device = torch.device('cpu')
    
    model = ImprovedPersonalityMLP(input_dim, output_dim).to(device)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model, scaler, label_encoder, device

# Complete questions
SCENARIO_QUESTIONS = [
    ("social_energy", "After spending the entire day with the team, I would feel energized and want to continue socializing rather than needing alone time."),
    ("alone_time_preference", "During free time at the retreat, I would prefer to find a quiet spot to be alone rather than join group activities."),
    ("talkativeness", "During team discussions and meals, I would be one of the most talkative people in the group."),
    ("deep_reflection", "I would spend time reflecting on the deeper meaning of the team-building exercises and how they relate to my personal growth."),
    ("group_comfort", "I would feel completely comfortable participating in all group activities, even with people I don't know well."),
    ("party_liking", "I would be excited about the evening social events and parties planned for the retreat."),
    ("listening_skill", "I would be good at listening to others' ideas and concerns during group discussions."),
    ("empathy", "I would easily understand and relate to how my teammates are feeling during challenging activities."),
    ("creativity", "I would come up with creative solutions during problem-solving activities."),
    ("organization", "I would help keep track of schedules, materials, and ensure everything runs smoothly."),
    ("leadership", "I would naturally take charge and guide the team during group challenges."),
    ("risk_taking", "I would be willing to try new, potentially challenging activities even if they seem risky."),
    ("public_speaking_comfort", "I would be comfortable giving presentations or speaking in front of the entire group."),
    ("curiosity", "I would be eager to learn about new activities and ask lots of questions about the retreat program."),
    ("routine_preference", "I would prefer to have a structured schedule with planned activities rather than spontaneous, free-form time."),
    ("excitement_seeking", "I would actively seek out the most exciting and adventurous activities available."),
    ("friendliness", "I would make an effort to be friendly and approachable to everyone at the retreat."),
    ("emotional_stability", "I would remain calm and composed even during stressful or challenging situations."),
    ("planning", "I would plan ahead for the retreat, researching activities and preparing accordingly."),
    ("spontaneity", "I would be open to changing plans and going with the flow when unexpected opportunities arise."),
    ("adventurousness", "I would be excited to try outdoor activities, new sports, or other adventurous experiences."),
    ("reading_habit", "During downtime, I would prefer to read a book rather than engage in social activities."),
    ("sports_interest", "I would be enthusiastic about participating in sports and physical activities."),
    ("online_social_usage", "I would frequently check social media and stay connected online during the retreat."),
    ("travel_desire", "I would be excited about the travel aspect and see this as an opportunity to explore a new place."),
    ("gadget_usage", "I would rely heavily on my phone, laptop, and other technology during the retreat."),
    ("work_style_collaborative", "I would prefer to work closely with others on team challenges rather than tackle problems individually."),
    ("decision_speed", "I would make quick decisions during group activities without needing much time to think."),
    ("stress_handling", "I would handle any conflicts, delays, or unexpected challenges at the retreat with ease.")
]

def initialize_session_state():
    """Initialize session state variables"""
    if "answers" not in st.session_state:
        st.session_state.answers = {}
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    if "assessment_started" not in st.session_state:
        st.session_state.assessment_started = False
    if "results_ready" not in st.session_state:
        st.session_state.results_ready = False

def create_progress_bar(current: int, total: int) -> None:
    """Create progress bar"""
    progress = current / total
    st.markdown(f"""
    <div class="progress-text">
        Question {current + 1} of {total} ({progress:.0%} Complete)
    </div>
    """, unsafe_allow_html=True)
    st.progress(progress)

def display_metric_card(title: str, value: str, color: str) -> str:
    """Generate metric card HTML"""
    return f"""
    <div class="metric-card">
        <h3>{title}</h3>
        <h2 style="color: {color} !important;">{value}</h2>
    </div>
    """

def create_results_visualization(probabilities: np.ndarray, labels: List[str], predicted_label: str) -> None:
    """Create results visualization with Personality Mirror chatbot"""
    
    # Results header
    st.markdown(f"""
    <div class="results-header">
        <h1>🎉 Your Personality Assessment Results</h1>
        <h2>🌟 Primary Type: {predicted_label.upper()}</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # PERSONALITY MIRROR CHATBOT
    create_chatbot_interface()
    
    # Probability chart
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=probabilities * 100,
            marker_color=px.colors.qualitative.Set3[:len(labels)],
            text=[f'{p:.1%}' for p in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Personality Type Probability Distribution",
        title_font_color="white",
        xaxis_title="Personality Types",
        yaxis_title="Confidence (%)",
        xaxis_title_font_color="white",
        yaxis_title_font_color="white",
        xaxis_tickfont_color="white",
        yaxis_tickfont_color="white",
        yaxis=dict(range=[0, 100]),
        height=500,
        template="plotly_dark",
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a'
    )
    
    # FIXED: Updated to use width='stretch' instead of use_container_width=True
    st.plotly_chart(fig, width='stretch')
    
    # Metric cards
    col1, col2, col3 = st.columns(3)
    max_prob = np.max(probabilities)
    
    with col1:
        st.markdown(display_metric_card("Confidence Level", f"{max_prob:.1%}", "#1f77b4"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(display_metric_card("Primary Trait", predicted_label, "#17becf"), unsafe_allow_html=True)
    
    with col3:
        second_idx = np.argsort(probabilities)[-2]
        second_label = labels[second_idx]
        second_prob = probabilities[second_idx]
        st.markdown(display_metric_card("Secondary Trait", f"{second_label}<br/><small>{second_prob:.1%}</small>", "#ff7f0e"), unsafe_allow_html=True)

def get_interpretation(max_prob: float) -> str:
    """Generate interpretation"""
    if max_prob > 0.7:
        return ("🎯 Strong Profile Match: Your personality type is very clear and well-defined. "
                "Your responses reveal consistent behaviors and preferences, indicating a strong alignment "
                "with this trait. Embrace these strengths as they shape your natural way of engaging with the world.")
    elif max_prob > 0.5:
        return ("⚖️ Moderate Profile: You exhibit a balance between different personality traits. "
                "This suggests flexibility and adaptability, enabling you to navigate a variety of situations "
                "effectively. Your personality may shift subtly depending on context, showing both strengths and areas of growth.")
    else:
        return ("🌈 Balanced Profile: Your personality displays a rich blend of diverse traits. "
                "This versatility allows you to relate to many kinds of people and situations. "
                "Such balance can be a significant asset in dynamic environments requiring varied approaches.")

def main():
    """Main application"""
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 class="main-title">🎯 AI-Powered Personality Assessment</h1>
        <h3 class="main-subtitle">Team Retreat Edition</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.assessment_started:
        # Scenario
        st.markdown("""
        <div class="scenario-box">
            <h2>🏔️ Mountain Retreat Adventure</h2>
            <p>You've been invited to an exciting weekend team retreat at a mountain resort! 
            You'll spend three days with 20 colleagues mixing outdoor adventures (hiking, rock climbing, team challenges), creative workshops, and social events around a campfire.
            You'll stay in shared cabins with a mix of familiar faces and new colleagues - from outgoing social butterflies to quieter team members. 
            The weekend includes everything from adrenaline-pumping activities to collaborative problem-solving and relaxed social time.
            Imagine yourself in this scenario and answer honestly based on how you'd naturally behave. 
            There are no right or wrong answers - just be genuine about your preferences and reactions!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="feature-box">
                <h4>⏱️ Duration</h4>
                <p>10-15 minutes</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="feature-box">
                <h4>🧠 AI Analysis</h4>
                <p>Neural network insights</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="feature-box">
                <h4>🪞 Talk to your mirror</h4>
                <p>Ask anything—get real insight</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🚀 Start Assessment", key="start_btn"):
            st.session_state.assessment_started = True
            st.rerun()
    
    elif not st.session_state.results_ready:
        # Questions
        create_progress_bar(st.session_state.current_question, len(SCENARIO_QUESTIONS))
        
        current_q = SCENARIO_QUESTIONS[st.session_state.current_question]
        question_id, question_text = current_q
        
        st.markdown(f"""
        <div class="question-box">
            <h3>Question {st.session_state.current_question + 1}</h3>
            <p>{question_text}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # FIXED: Get current answer for this specific question
        current_answer = st.session_state.answers.get(question_id, 5)
        
        # FIXED: Use unique key for each question to prevent slider confusion
        answer = st.slider(
            "Your Response",
            min_value=1,
            max_value=10,
            value=current_answer,
            key=f"slider_q{st.session_state.current_question}_{question_id}",  # Unique key per question
            help="1 = Strongly Disagree | 5 = Neutral | 10 = Strongly Agree"
        )
        
        # Store the answer for this question
        st.session_state.answers[question_id] = answer
        
        labels = ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Slightly Disagree", 
                 "Neutral", "Slightly Agree", "Somewhat Agree", "Agree", "Strongly Agree", "Completely Agree"]
        
        st.markdown(f"""
        <div class="selection-info">
            <p><strong>Your selection:</strong> {answer}/10 - {labels[answer-1]}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.current_question > 0:
                if st.button("⬅️ Previous", key="prev_btn"):
                    st.session_state.current_question -= 1
                    st.rerun()
        
        with col3:
            if st.session_state.current_question < len(SCENARIO_QUESTIONS) - 1:
                if st.button("Next ➡️", key="next_btn"):
                    st.session_state.current_question += 1
                    st.rerun()
            else:
                if st.button("✅ Complete Assessment", key="submit_btn"):
                    with st.spinner("🧠 Analyzing your responses..."):
                        calculate_and_display_results()
    
    else:
        display_final_results()

def calculate_and_display_results():
    """Calculate results"""
    try:
        model, scaler, label_encoder, device = load_and_train_model()
        
        user_answers = []
        for feature, _ in SCENARIO_QUESTIONS:
            user_answers.append(st.session_state.answers.get(feature, 5))
        
        if len(user_answers) != scaler.n_features_in_:
            user_answers = user_answers[:scaler.n_features_in_]
            while len(user_answers) < scaler.n_features_in_:
                user_answers.append(5)
        
        user_data = np.array(user_answers).reshape(1, -1)
        user_scaled = scaler.transform(user_data)
        user_tensor = torch.tensor(user_scaled, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            logits = model(user_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_index = np.argmax(probabilities)
            predicted_label = label_encoder.inverse_transform([predicted_index])[0]
        
        st.session_state.prediction_results = {
            'predicted_label': predicted_label,
            'probabilities': probabilities,
            'labels': label_encoder.classes_
        }
        st.session_state.results_ready = True
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing results: {str(e)}")
        if st.button("🔄 Try Again"):
            st.session_state.results_ready = False
            st.rerun()

def display_final_results():
    """Display final results"""
    results = st.session_state.prediction_results
    
    create_results_visualization(
        results['probabilities'],
        results['labels'],
        results['predicted_label']
    )
    
    max_prob = np.max(results['probabilities'])
    interpretation = get_interpretation(max_prob)
    
    st.markdown(f"""
    <div class="interpretation-box">
        <h3>💡 Interpretation</h3>
        <p>{interpretation}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed breakdown
    st.markdown("""
    <h3 class="results-section-header">📊 Detailed Breakdown</h3>
    """, unsafe_allow_html=True)
    
    breakdown_df = pd.DataFrame({
        'Personality Type': results['labels'],
        'Confidence': [f"{prob:.1%}" for prob in results['probabilities']],
        'Score': results['probabilities']
    }).sort_values('Score', ascending=False)
    
    # FIXED: Updated to use width='stretch' instead of use_container_width=True
    st.dataframe(breakdown_df[['Personality Type', 'Confidence']], width='stretch')
    
    # Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retake Assessment", key="retake_btn"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        results_text = f"""Personality Assessment Results
=============================
Primary Type: {results['predicted_label']}
Confidence: {max_prob:.1%}

Detailed Breakdown:
{chr(10).join([f"{label}: {prob:.1%}" for label, prob in zip(results['labels'], results['probabilities'])])}

Interpretation: {interpretation}
        """
        st.download_button(
            label="📄 Download Results",
            data=results_text,
            file_name="personality_results.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

